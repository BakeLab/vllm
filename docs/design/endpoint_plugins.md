# Endpoint Plugins

This fork serves HTTP and gRPC through the Rust frontend. It does not load
`vllm.endpoint_plugins` or attach Python FastAPI routers.

`VLLM_PLUGINS` still controls the supported engine plugin groups described in
[Plugin System](plugin_system.md). Python `vllm run-batch` retains its local
chat completion, embedding, scoring, transcription, and translation handlers;
it does not start an HTTP server or attach endpoint plugins.
