"""Entry point of the Hugging Face Space (Gradio SDK on ZeroGPU). See deploy/README.md."""

import spaces  # noqa: F401  -- first, so ZeroGPU is set up before anything touches CUDA

from lecturelens.web.space import create_space_from_env, launch

app = create_space_from_env()

if __name__ == "__main__":
    launch(app)
