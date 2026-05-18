# Smart Turn v3 model

Store `smart-turn-v3.0.onnx` in a model artifact location and set the absolute
path explicitly in the node/system config as `backend.model_path`.

The node does not discover package-local models and does not use this directory
as a fallback. It refuses to start when the configured model is missing or when
the file is a Git LFS pointer instead of the actual ONNX payload.
