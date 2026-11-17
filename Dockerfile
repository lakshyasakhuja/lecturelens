# The Slideseek web app: the search API and the frontend, on a CPU.
#
#   docker build -t slideseek .
#   docker run -p 7860:7860 -v "$PWD/index:/home/user/index" slideseek
#
# The index is not baked into the image. Mount one at /home/user/index, or set
# SLIDESEEK_INDEX_REPO to a Hugging Face dataset repository and it is downloaded at
# start-up (this is how the hosted demo gets its index).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run containers as user 1000; create it and work in its home.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# CPU-only PyTorch first: the default wheel drags in several gigabytes of CUDA libraries
# that a CPU host can never use.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user slideseek ./slideseek
RUN pip install --user ".[models,ocr,web]"

# Fetch the two query encoders at build time, so a cold start does not begin with a
# three-gigabyte download.
RUN python -c "from huggingface_hub import snapshot_download as get; get('Qwen/Qwen3-Embedding-0.6B'); get('google/siglip2-base-patch16-224')"

COPY --chown=user config ./config

ENV SLIDESEEK_INDEX=/home/user/index \
    SLIDESEEK_DEVICE=cpu \
    HF_HUB_OFFLINE=0

EXPOSE 7860
# No access log: a request line contains the search text, and queries are nobody's business.
CMD ["uvicorn", "slideseek.web.server:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
