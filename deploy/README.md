# Deploying the web app

The app is one container: the search API and the static frontend, on a CPU. It needs about
6 GB of memory for the two query encoders. It does not need a GPU.

## Locally

```bash
uv run lecturelens serve --index index
```

Add `--videos videos` to play your own recordings in the page instead of linking out.

## With Docker

```bash
docker build -t lecturelens .
```

```bash
docker run -p 7860:7860 -v "$PWD/index:/home/user/index" lecturelens
```

## On Hugging Face Spaces

The hosted demo is a Docker Space on the free CPU tier. The index lives in a separate
dataset repository so that it can be rebuilt without redeploying the app.

1. Publish the index (after `lecturelens index`):

   ```bash
   uv run lecturelens push <user>/lecturelens-index --index index
   ```

2. Create a Space with the **Docker** SDK, then add a variable under
   *Settings → Variables and secrets*:

   | Name | Value |
   |---|---|
   | `LECTURELENS_INDEX_REPO` | `<user>/lecturelens-index` |

3. Push this repository to the Space, with `deploy/space/README.md` as the Space's
   `README.md` (it carries the metadata Spaces read: SDK, port, title):

   ```bash
   git clone https://huggingface.co/spaces/<user>/lecturelens space
   ```

   ```bash
   rsync -a --delete --exclude .git --exclude videos --exclude work --exclude index --exclude .venv ./ space/ && cp deploy/space/README.md space/README.md
   ```

   ```bash
   cd space && git add -A && git commit -m "Deploy LectureLens" && git push
   ```

The Space builds the image, downloads the index on first start, and serves on port 7860.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LECTURELENS_INDEX` | `index` | Directory holding the index. |
| `LECTURELENS_INDEX_REPO` | unset | Dataset repository to download the index from when the directory is empty. |
| `LECTURELENS_VIDEOS` | unset | Directory of local recordings to play in the page. |
| `LECTURELENS_EXAMPLES` | `config/examples.json` | Example questions shown under the search box. |
| `LECTURELENS_BENCHMARK` | `results/test/metrics.json` | Benchmark metrics to show; the page shows none if the file is absent. |
| `LECTURELENS_DEVICE` | `cpu` | Device for the query encoders. |

## Why not Gradio

The first plan was a Gradio app on a ZeroGPU Space. Search turned out to need no GPU at
all (about 100 ms a query on a CPU, both encoders included), and a hand-written page gives
full control over the design, so the app is plain FastAPI with a static frontend instead.
The one thing that would want a GPU, the optional cross-encoder reranker, is available from
the CLI and the evaluation scripts but is not exposed in the hosted demo.
