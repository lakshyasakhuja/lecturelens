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
- **"Rerank on a GPU"** is built but switched off, because on ZeroGPU it does not work yet;
  [what was found](#what-the-first-deployment-showed) is below. With
  `LECTURELENS_RERANK=1` the page shows a switch that rescores the fused top 20 with the
  cross-encoder inside a `@spaces.GPU` call, on the visitor's own daily ZeroGPU allowance
  (2 minutes without an account, 5 with a free one). If the call fails, the page keeps the
  ordinary results and says so.

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

To run the Space's app on your own machine, with the switch, where `@spaces.GPU` does
nothing and the reranker runs on the CPU:

```bash
uv sync --extra index --extra web --extra space
```

```bash
PYTHONPATH=. LECTURELENS_INDEX=index LECTURELENS_RERANK=1 uv run python deploy/space/app.py
```

### What the first deployment showed

Deployed on 17 November 2026 to a free account (torch 2.13.0, `spaces` 0.51.3, Gradio
6.32.0):

- `lecturelens deploy` was given ZeroGPU hardware through the API, the Space built and
  started, and the page and the search API work there. A search takes about 75 ms.
- **Reranking fails.** Every call ends in ZeroGPU's worker with `RuntimeError: No CUDA GPUs
  are available`, raised from `spaces/zero/wrappers.py` in `worker_init`, before any of
  this project's code runs on the GPU.
- The cause was narrowed down on the Space itself, with a one-line `@spaces.GPU` function
  called after each start-up step. The GPU call works under Gradio's server mode, and
  keeps working after importing the libraries, loading the index, loading and running
  SigLIP, and loading the Qwen3 text encoder. It stops working, for the life of the
  process, after the Qwen3 text encoder's **first forward pass on the CPU**, through
  sentence-transformers or through `transformers` alone. Which call inside that forward
  pass is responsible was not found.
- A [forum thread](https://discuss.huggingface.co/t/runtimeerror-no-cuda-gpus-are-available-and-valueerror/180008)
  reports the same error after CPU work in the Space's main process, and that doing the
  work in a separate process avoids it.

Searching needs that forward pass, so reranking is off by default rather than offered and
broken. The fix to try next is to run the query encoders in a process of their own, so the
process that ZeroGPU forks its workers from never runs them. It has not been tried.

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
| `LECTURELENS_RERANK` | unset | `1` offers "Rerank on a GPU" in the Space's app. Not read by the container. |

## Why the frontend is not Gradio's

Gradio's components would have been the short way to a Space. Search needs no GPU (about
100 ms a query on a CPU, both encoders included), and a hand-written page gives full control
over the design, so the app is plain FastAPI with a static frontend, and Gradio supplies
only what ZeroGPU requires: the server and the queue in front of the one GPU call. When
reranking is offered, the page loads Gradio's JavaScript client, from this origin and only
once the switch is on, because that client is what tells ZeroGPU whose allowance a call
counts against.
