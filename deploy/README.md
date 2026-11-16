# Deploying the web app

The app is the search API and the static frontend in one process, on a CPU. With both query
encoders loaded it used about 4 GB of memory in a local container run, so a host with 8 GB
or more is a comfortable fit and one with 4 GB is at the limit. Searching does not need a
GPU.

There are two ways to run it, from the same code:

| | Runs as | Cost |
|---|---|---|
| [A Hugging Face Space on ZeroGPU](#on-hugging-face-spaces-free) | A Gradio app, `deploy/space/app.py` | Free |
| [A container](#with-docker), anywhere | Plain FastAPI under uvicorn, `Dockerfile` | Whatever 8 GB of memory costs at the host |

## Locally

```bash
uv run lecturelens serve --index index
```

Add `--videos videos` to play your own recordings in the page instead of linking out.

## On Hugging Face Spaces, free

Hugging Face lets a free personal account host up to two Spaces on **ZeroGPU**, shared
hardware that attaches a GPU to a Space only for the length of one function call. ZeroGPU
runs Gradio apps only, so on a Space the app is served by Gradio's server mode
(`gradio.Server`): a FastAPI application with Gradio's queue attached. The page, the JSON
API and the search itself are exactly the ones `lecturelens serve` runs.

- **Searching** runs on the Space's CPU and costs a visitor nothing.
- **"Rerank on a GPU"**, a switch the page shows only on a Space, rescores the fused top 20
  with the cross-encoder inside a `@spaces.GPU` call. That uses a second or two of the
  visitor's own daily ZeroGPU allowance (2 minutes without an account, 5 with a free one).
  If no GPU is free or the allowance is spent, the page keeps the ordinary results and says
  so.

What the account needs, as the
[ZeroGPU documentation](https://huggingface.co/docs/hub/en/spaces-zerogpu) read in November
2026: a verified email address and an account more than 30 days old. Check it before
relying on it.

1. Log in once. This stores a token on your machine; create one with *write* access at
   <https://huggingface.co/settings/tokens>.

   ```bash
   uv run hf auth login
   ```

2. Publish the index, as a public dataset repository with an attribution card:

   ```bash
   uv run lecturelens push <user>/lecturelens-index --index index
   ```

3. Create the Space, point it at the index and upload the app:

   ```bash
   uv run lecturelens deploy <user>/lecturelens --index-repo <user>/lecturelens-index
   ```

The Space then builds for a few minutes, downloads the index, loads the models and starts.
Its *Logs* tab shows each step. Run step 3 again to update the Space after changing the
code, and step 2 after rebuilding the index.

`lecturelens deploy` uploads the package's source, `config/examples.json`, the measured
benchmark if `results/test/metrics.json` exists, and the three files in `deploy/space`:
`app.py` (the entry point), `requirements.txt` and `README.md` (the Space's metadata: SDK,
Gradio version, Python version, models to fetch at build time). It uploads no video and no
index. The same thing by hand: create a Space with the Gradio SDK and ZeroGPU hardware, add
the variable `LECTURELENS_INDEX_REPO` under *Settings → Variables and secrets*, and push
those files to it.

To run the Space's app on your own machine, where `@spaces.GPU` does nothing and the
reranker runs on the CPU:

```bash
uv sync --extra index --extra web --extra space
```

```bash
PYTHONPATH=. LECTURELENS_INDEX=index uv run python deploy/space/app.py
```

### What has and has not been tested

The Space's app has been run on a laptop against the real index: the page, the API, and
reranking through Gradio's JavaScript client. The folder `lecturelens deploy` uploads has
been run on its own. The unit tests launch the Gradio server and call the reranking
endpoint. What cannot be run anywhere but on Hugging Face is ZeroGPU itself, which patches
PyTorch and needs a CUDA build of it, so the first deployment is the first real test of
the GPU call. Three things were done to keep to the documented path: `spaces` is imported
before anything else, every model is loaded before the server starts, and PyTorch is
pinned to a release on ZeroGPU's supported list.

## With Docker

```bash
docker build -t lecturelens .
```

```bash
docker run -p 7860:7860 -v "$PWD/index:/home/user/index" lecturelens
```

Any host that runs a Docker image works: set `LECTURELENS_INDEX_REPO` (or mount an index at
`/home/user/index`) and route traffic to port 7860. Memory is what decides the plan. Free
and entry-level instances with 512 MB to 2 GB cannot load the encoders; pick one with 8 GB.

The image also runs as a Docker Space on Hugging Face, on CPU Basic hardware (2 vCPU,
16 GB), with `sdk: docker` and `app_port: 7860` in the Space's `README.md`. Creating a
Docker Space on a personal account needs a PRO plan, which is why the free route above is a
Gradio Space instead. The container does not offer reranking.

## Configuration

Both ways of running read the same environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `LECTURELENS_INDEX` | `index` | Directory holding the index. |
| `LECTURELENS_INDEX_REPO` | unset | Dataset repository to download the index from when the directory is empty. |
| `LECTURELENS_VIDEOS` | unset | Directory of local recordings to play in the page. Not read on a Space. |
| `LECTURELENS_EXAMPLES` | `config/examples.json` | Example questions shown under the search box. |
| `LECTURELENS_BENCHMARK` | `results/test/metrics.json` | Benchmark metrics to show; the page shows none if the file is absent. |
| `LECTURELENS_DEVICE` | `cpu` | Device for the query encoders. Always `cpu` on ZeroGPU. |

## Why the frontend is not Gradio's

Gradio's components would have been the short way to a Space. Search needs no GPU (about
100 ms a query on a CPU, both encoders included), and a hand-written page gives full control
over the design, so the app is plain FastAPI with a static frontend, and Gradio supplies
only what ZeroGPU requires: the server and the queue in front of the one GPU call. The page
loads Gradio's JavaScript client, from this origin and only when reranking is switched on,
because that client is what tells ZeroGPU whose allowance a call counts against.
