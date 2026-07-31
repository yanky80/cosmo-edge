# CosmoEdge Platform Terms

- Target platform: the deployment hardware and OS profile selected at build time, such as `x86`, `sophon`, or `rk3588`.
- Inference backend: the runtime that loads and executes model artifacts, such as ONNX Runtime, BMRuntime, or RKNN Runtime.
- Media backend: the decode, image-processing, and encode implementation paired with a target platform.
- Model artifact: a single backend-loadable model file such as `.onnx`, `.nn`, `.bmodel`, or `.rknn`.
- Model package: `config.json`, one target platform's model artifacts, and any helper files imported together.
- Static platform build: one build output contains exactly one target platform profile and does not load vendor backends as runtime plugins.
