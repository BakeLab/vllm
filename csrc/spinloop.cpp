#include <Python.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#if defined(__linux__)
  #include <linux/futex.h>
  #include <sys/syscall.h>
  #include <unistd.h>
#endif

extern "C" {

#include <stdbool.h>
#include <time.h>

#if defined(__i386__) || defined(__x86_64__)
  #include <cpuid.h>
  #include <x86intrin.h>
#endif

#if defined(CLOCK_MONOTONIC_RAW)
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC_RAW
#else
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC
#endif

#define CPU_SUPPORT_NONE 0
#define CPU_SUPPORT_MONITORX 1

#define MWAITX_DEFAULT_TIMEOUT_CYCLES 1000000

typedef struct {
  unsigned int cpu_support;
  unsigned int max_monitor_line_size;
} spinloop_state_t;

static void determine_cpu_support(spinloop_state_t* state) {
  state->cpu_support = CPU_SUPPORT_NONE;
  state->max_monitor_line_size = 0;

#if defined(__i386__) || defined(__x86_64__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 1) {
    // AMD CPU (possible monitorx/mwaitx support)
    if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163) {
      if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 1 &&
          eax >= 0x80000001 &&
          __get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 1) {
        if ((ecx & (1 << 29)) != 0) {
          state->cpu_support = CPU_SUPPORT_MONITORX;
        }
      }
    }
  }

  if (state->cpu_support == CPU_SUPPORT_MONITORX) {
    if (__get_cpuid(5, &eax, &ebx, &ecx, &edx) == 1) {
      state->max_monitor_line_size = ebx & 0xff;
    }
  }
#endif
}

static PyObject* method_spinloop(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  Py_buffer buffer;
  PyObject* callback;
  double timeout = 0.;

  spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(self);
  if (state == NULL) {
    PyErr_SetString(PyExc_TypeError, "Failed to retrieve module state!");
    return NULL;
  }

  static const char* keywords[] = {"buffer", "callback", "timeout", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*O|d", (char**)keywords,
                                   &buffer, &callback, &timeout)) {
    return NULL;
  }

  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "callback parameter must be callable!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  struct timespec t_start;
  if (clock_gettime(TIMEOUT_CLOCK, &t_start) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  bool result = false;
  bool error = false;
  bool have_timeout = (timeout > 1e-9);
  unsigned int iteration = 0;
  const bool buffer_qualifies = (buffer.len <= state->max_monitor_line_size);

  while (true) {
    PyObject* res = PyObject_CallNoArgs(callback);
    if (res == NULL) {
      error = true;
      break;
    }
    int ok = (res == Py_True);
    Py_DECREF(res);

    if (ok) {
      result = true;
      break;
    }

    // Check timeout at most every 16 iterations to avoid clock_gettime and
    // comparison cost
    if (have_timeout && (iteration & 15u) == 0) {
      struct timespec t_now;
      if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
        error = true;
        break;
      }

      const double elapsed = (double)(t_now.tv_sec - t_start.tv_sec) +
                             (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
      if (elapsed >= timeout) {
        result = false;
        break;
      }
    }
    ++iteration;

#if defined(__i386__) || defined(__x86_64__)
    // monitorx + mwaitx with qualified buffer
    if (buffer_qualifies && state->cpu_support == CPU_SUPPORT_MONITORX) {
      _mm_monitorx(buffer.buf, 0, 0);

      // Check once more in case the buffer has been modified while we were
      // arming the monitor hardware
      res = PyObject_CallNoArgs(callback);
      if (res == NULL) {
        error = true;
        break;
      }
      ok = (res == Py_True);
      Py_DECREF(res);

      if (ok) {
        result = true;
        break;
      }

      // Run mwaitx with enabled timeout (bit 1). The actual timeout value
      // is not very important, we just want to ensure we don't lock up
      // here for too long.
      Py_BEGIN_ALLOW_THREADS _mm_mwaitx((1 << 1), 0,
                                        MWAITX_DEFAULT_TIMEOUT_CYCLES);
      Py_END_ALLOW_THREADS
    }

    // Fallback: Busy poll
    else {
#endif
      // Give other threads a chance to be scheduled
      Py_BEGIN_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield" :: : "memory");
#endif
      Py_END_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
    }
#endif
  }

  PyBuffer_Release(&buffer);

  if (error) {
    return NULL;
  }

  if (result) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

#if defined(__linux__) && defined(SYS_futex_waitv) && defined(FUTEX_32)
static int get_word(PyObject* obj, Py_buffer* view) {
  if (PyObject_GetBuffer(obj, view, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) < 0)
    return -1;
  if (view->len < 4 || ((uintptr_t)view->buf % alignof(uint32_t))) {
    PyBuffer_Release(view);
    PyErr_SetString(PyExc_ValueError,
                    "word buffer must contain an aligned writable uint32");
    return -1;
  }
  return 0;
}

static PyObject* publish(PyObject* self, PyObject* obj) {
  Py_buffer view;
  if (get_word(obj, &view) < 0) return NULL;
  uint32_t* word = static_cast<uint32_t*>(view.buf);
  __atomic_fetch_add(word, 1, __ATOMIC_RELEASE);
  long result = syscall(SYS_futex, word, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
  int error = errno;
  PyBuffer_Release(&view);
  if (result < 0) {
    errno = error;
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  Py_RETURN_NONE;
}

static PyObject* cancel_wait(PyObject* self, PyObject* obj) {
  Py_buffer view;
  if (get_word(obj, &view) < 0) return NULL;
  uint32_t* word = static_cast<uint32_t*>(view.buf);
  __atomic_store_n(word, 1, __ATOMIC_RELEASE);
  long result = syscall(SYS_futex, word, FUTEX_WAKE | FUTEX_PRIVATE_FLAG,
                        INT_MAX, NULL, NULL, 0);
  int error = errno;
  PyBuffer_Release(&view);
  if (result < 0) {
    errno = error;
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  Py_RETURN_NONE;
}

static uint64_t now_ns(void) {
  struct timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return (uint64_t)time.tv_sec * 1000000000ULL + time.tv_nsec;
}

static PyObject* wait_ready(PyObject* self, PyObject* args) {
  PyObject *ctrl_obj, *cancel_obj, *shm_obj;
  Py_ssize_t offset, read_offset;
  long long timeout_ms, spin_us;
  if (!PyArg_ParseTuple(args,
                        "OOO"
                        "nn"
                        "LL",
                        &ctrl_obj, &cancel_obj, &shm_obj, &offset, &read_offset,
                        &timeout_ms, &spin_us))
    return NULL;
  if (timeout_ms < -1 || spin_us < 0 || spin_us > LLONG_MAX / 1000 ||
      timeout_ms > LLONG_MAX / 1000000) {
    PyErr_SetString(PyExc_ValueError, "invalid timeout or spin budget");
    return NULL;
  }
  Py_buffer ctrl, cancel, shm;
  if (get_word(ctrl_obj, &ctrl) < 0) return NULL;
  if (get_word(cancel_obj, &cancel) < 0) {
    PyBuffer_Release(&ctrl);
    return NULL;
  }
  if (PyObject_GetBuffer(shm_obj, &shm, PyBUF_C_CONTIGUOUS) < 0) {
    PyBuffer_Release(&cancel);
    PyBuffer_Release(&ctrl);
    return NULL;
  }
  if (offset < 0 || read_offset <= 0 || offset >= shm.len ||
      read_offset >= shm.len - offset) {
    PyBuffer_Release(&shm);
    PyBuffer_Release(&cancel);
    PyBuffer_Release(&ctrl);
    PyErr_SetString(PyExc_ValueError, "metadata offsets outside shared buffer");
    return NULL;
  }
  uint32_t* seq = static_cast<uint32_t*>(ctrl.buf);
  uint32_t* cancelled = static_cast<uint32_t*>(cancel.buf);
  unsigned char* written = (unsigned char*)shm.buf + offset;
  unsigned char* read = written + read_offset;
  uint64_t start = now_ns();
  uint64_t spin_until = start + (uint64_t)spin_us * 1000;
  struct timespec deadline;
  struct timespec* deadline_ptr = NULL;
  if (timeout_ms >= 0) {
    uint64_t end = start + (uint64_t)timeout_ms * 1000000;
    deadline.tv_sec = end / 1000000000;
    deadline.tv_nsec = end % 1000000000;
    deadline_ptr = &deadline;
  }
  int status = 0, error = 0;
  Py_BEGIN_ALLOW_THREADS for (;;) {
    uint32_t expected = __atomic_load_n(seq, __ATOMIC_ACQUIRE);
    if (__atomic_load_n(cancelled, __ATOMIC_ACQUIRE)) {
      status = 1;
      break;
    }
    // Match slot reuse: read acknowledgement before the published flag.
    if (__atomic_load_n(read, __ATOMIC_ACQUIRE) == 0 &&
        __atomic_load_n(written, __ATOMIC_ACQUIRE) == 1) {
      status = 0;
      break;
    }
    uint64_t current = now_ns();
    if (deadline_ptr &&
        (current / 1000000000 > (uint64_t)deadline.tv_sec ||
         (current / 1000000000 == (uint64_t)deadline.tv_sec &&
          current % 1000000000 >= (uint64_t)deadline.tv_nsec))) {
      status = 2;
      break;
    }
    if (current < spin_until) {
  #if defined(__x86_64__) || defined(__i386__)
      __asm__ __volatile__("pause");
  #elif defined(__aarch64__)
      __asm__ __volatile__("yield");
  #endif
      continue;
    }
    struct futex_waitv waits[2] = {};
    waits[0].val = expected;
    waits[0].uaddr = (uintptr_t)seq;
    waits[0].flags = FUTEX_32;
    waits[1].uaddr = (uintptr_t)cancelled;
    waits[1].flags = FUTEX_32 | FUTEX_PRIVATE_FLAG;
    long result =
        syscall(SYS_futex_waitv, waits, 2, 0, deadline_ptr, CLOCK_MONOTONIC);
    if (result >= 0 || errno == EAGAIN) continue;
    if (errno == ETIMEDOUT) {
      status = 2;
      break;
    }
    if (errno == EINTR) {
      status = 3;
      break;
    }
    error = errno;
    break;
  }
  Py_END_ALLOW_THREADS PyBuffer_Release(&shm);
  PyBuffer_Release(&cancel);
  PyBuffer_Release(&ctrl);
  if (error) {
    errno = error;
    return PyErr_SetFromErrno(PyExc_OSError);
  }
  if (status == 3 && PyErr_CheckSignals() < 0) return NULL;
  return PyLong_FromLong(status);
}

static PyObject* native_supported(PyObject*, PyObject*) {
  uint32_t word = 0;
  struct futex_waitv waiter = {};
  waiter.val = 1;
  waiter.uaddr = (uintptr_t)&word;
  waiter.flags = FUTEX_32 | FUTEX_PRIVATE_FLAG;
  long result = syscall(SYS_futex_waitv, &waiter, 1, 0, NULL, CLOCK_MONOTONIC);
  return PyBool_FromLong(result >= 0 || errno == EAGAIN);
}
#else
static PyObject* native_supported(PyObject*, PyObject*) { Py_RETURN_FALSE; }
static PyObject* publish(PyObject*, PyObject*) {
  PyErr_SetString(PyExc_NotImplementedError, "futex_waitv is unavailable");
  return NULL;
}
static PyObject* cancel_wait(PyObject* self, PyObject* args) {
  return publish(self, args);
}
static PyObject* wait_ready(PyObject* self, PyObject* args) {
  return publish(self, args);
}
#endif

static PyMethodDef spinloop_methods[] = {
    {"native_publish", publish, METH_O,
     "Publish shared notification sequence."},
    {"native_cancel", cancel_wait, METH_O, "Cancel a local native waiter."},
    {"native_wait", wait_ready, METH_VARARGS,
     "Wait for shared metadata readiness."},
    {"native_supported", native_supported, METH_NOARGS,
     "Probe futex_waitv support."},
    {"spinloop", (PyCFunction)method_spinloop, METH_VARARGS | METH_KEYWORDS,
     "Wait for store with callback"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef spinloop_module = {
    PyModuleDef_HEAD_INIT, "spinloop",
    "Hardware-optimized spinloops for Python", sizeof(spinloop_state_t),
    spinloop_methods};

PyMODINIT_FUNC PyInit_spinloop(void) {
  PyObject* m = PyModule_Create(&spinloop_module);
  if (m != NULL) {
    spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(m);
    if (state != NULL) {
      determine_cpu_support(state);
    }
  }
  return m;
}

}  // extern "C"
