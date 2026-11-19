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
uv run slideseek serve --index index
```

Add `--videos videos` to play your own recordings in the page instead of linking out.
Searching needs an account: sign up on the page the first time. Accounts are kept in
`accounts.db` in the folder the server runs from, which git ignores; see [Accounts](#accounts).

To let people add their own lectures from the page ([how it works](#peoples-own-lectures)),
fetch the browser's speech-recognition scripts once and name a folder to keep the lectures
in:

```bash
uv run slideseek vendor
```

```bash
uv run slideseek serve --index index --library libraries
```

## On Hugging Face Spaces, free

Hugging Face lets a free personal account host up to two Spaces on **ZeroGPU**, shared
hardware that attaches a GPU to a Space only for the length of one function call. ZeroGPU
runs Gradio apps only, so on a Space the app is served by Gradio's server mode
(`gradio.Server`): a FastAPI application with Gradio's queue attached. The page, the JSON
API and the search itself are exactly the ones `slideseek serve` runs.

- **Searching** runs on the Space's CPU and costs a visitor nothing, once they have an
  account. A Space's disk does not last, so the accounts are kept with a copy in a private
  dataset: see [Accounts](#accounts).
- **Adding your own lectures** is on when the Space is deployed with `--libraries-repo`
  (step 7 below). The visitor's browser does the speech recognition, so the Space's CPU
  only reads the slides and makes them searchable: see
  [People's own lectures](#peoples-own-lectures).
- **"Rerank on a GPU"** is built but switched off, because on ZeroGPU it does not work yet;
  [what was found](#what-the-first-deployment-showed) is below. With
  `SLIDESEEK_RERANK=1` the page shows a switch that rescores the fused top 20 with the
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
   uv run slideseek push <user>/slideseek-index --index index
   ```

3. Create the private dataset that keeps a copy of the accounts ([why](#accounts)):

   ```bash
   uv run hf repos create <user>/slideseek-accounts --type dataset --private
   ```

   Then create a token that can write to that dataset and nothing else: at
   <https://huggingface.co/settings/tokens>, choose *Fine-grained*, add the dataset under
   *Repositories permissions*, tick the read and the write permission for the selected
   repositories, and leave every other box empty.

4. Create the Space, point it at the index and the dataset, and upload the app:

   ```bash
   uv run slideseek deploy <user>/slideseek --index-repo <user>/slideseek-index --accounts-repo <user>/slideseek-accounts
   ```

5. On the Space, under *Settings → Variables and secrets*, add a **secret** named
   `SLIDESEEK_ACCOUNTS_TOKEN` holding the token from step 3, then *Restart this Space*.
   On a Space that already exists, add it before step 4 instead. Until the secret is
   there the Space does not start, and its *Logs* tab says it cannot reach the dataset.

6. Once the Space runs, sign up on it with the username you want to be the default admin,
   then deploy again with `--admin <username>` added to the command in step 4 ([why
   then](#admins)). The account is made an admin when the Space restarts.

7. To let people add their own lectures, deploy with a second private dataset for them
   ([why](#peoples-own-lectures)):

   ```bash
   uv run slideseek deploy <user>/slideseek --index-repo <user>/slideseek-index --accounts-repo <user>/slideseek-accounts --libraries-repo <user>/slideseek-libraries
   ```

   It fetches the browser's scripts (`slideseek vendor`), creates the dataset as private if
   it does not exist, and uploads the scripts with the app. The Space writes to the dataset
   with `SLIDESEEK_LIBRARIES_TOKEN` if that secret is set, and otherwise with the accounts'
   token, so either add this dataset to the fine-grained token from step 3 or give it a
   token of its own.

The Space then builds for a few minutes, downloads the index and the accounts, loads the
models and starts. Its *Logs* tab shows each step. Run step 4 again to update the Space
after changing the code, and step 2 after rebuilding the index.

`slideseek deploy` uploads the package's source, `config/examples.json`, the measured
benchmark if `results/test/metrics.json` exists, and the three files in `deploy/space`:
`app.py` (the entry point), `requirements.txt` and `README.md` (the Space's metadata: SDK,
Gradio version, Python version, models to fetch at build time). It uploads no video and no
index. The same thing by hand: create a Space with the Gradio SDK and ZeroGPU hardware, add
the variables `SLIDESEEK_INDEX_REPO`, `SLIDESEEK_ACCOUNTS_REPO` and `SLIDESEEK_ADMIN` under
*Settings → Variables and secrets*, and push those files to it.

To run the Space's app on your own machine, with the switch, where `@spaces.GPU` does
nothing and the reranker runs on the CPU:

```bash
uv sync --extra index --extra web --extra space
```

```bash
PYTHONPATH=. SLIDESEEK_INDEX=index SLIDESEEK_RERANK=1 uv run python deploy/space/app.py
```

### What the first deployment showed

Deployed on 17 November 2026 to a free account (torch 2.13.0, `spaces` 0.51.3, Gradio
6.32.0):

- `slideseek deploy` was given ZeroGPU hardware through the API, the Space built and
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

## Accounts

Every search needs an account. A visitor signs up with their name, a username, an email
address, a phone number with its country code, and a password, and is signed in by a
session cookie. What is checked and how it is kept is described in `slideseek/accounts.py`
and `slideseek/web/auth.py`. In short:

- Usernames and email addresses are unique, ignoring case.
- A password needs 8 characters, an uppercase letter, a lowercase letter, a number and a
  special character. It is stored only as a salted Argon2id hash.
- A phone number is checked for its country and stored in E.164 form (`+61412345678`).
- A session token is 256 random bits; the database holds only its SHA-256. The cookie is
  HttpOnly, and over HTTPS `Secure; SameSite=None; Partitioned`, so that signing in works
  inside the frame huggingface.co shows a Space in. Requests that change something are
  refused unless they come from the app's own page (`Sec-Fetch-Site`, or `Origin`).
- Failed sign-ins are limited per address and per account, and sign-ups per address.
- A visitor can sign out, and can delete their account with its password.
- Admins can see and change every account, at `/admin` ([Admins](#admins)).

Where the accounts are kept:

| Where it runs | What to set |
|---|---|
| A laptop | Nothing. They go in `accounts.db` where the server runs. |
| A container | `SLIDESEEK_DATABASE_URL`: a PostgreSQL URL, or SQLite on a mounted volume (see the `Dockerfile`). |
| A Hugging Face Space | `SLIDESEEK_ACCOUNTS_REPO`, a private dataset, and the secret `SLIDESEEK_ACCOUNTS_TOKEN` (steps 3 to 5 [above](#on-hugging-face-spaces-free)). |

### On a Space: a copy in a private dataset

A Space's own disk is wiped every time it restarts, is redeployed or wakes from sleep, and
accounts kept only on it go with it. Hugging Face's persistent storage would keep them, but
it is paid. So the accounts stay in a SQLite file on the Space's disk, and a copy of that
file is kept in a private dataset repository on the Hub, which the Space reaches over
HTTPS (`slideseek/backup.py`):

- **At start-up** the copy is downloaded before the database is opened. If the dataset has
  a copy that cannot be downloaded after four tries, or that is damaged, the Space does not
  start: starting with no accounts would overwrite the copy at the first change. The
  *Logs* tab says what went wrong.
- **After a change** (a sign-up, a sign-in, a sign-out, a deletion) a snapshot of the file
  is uploaded: 10 seconds after the first change, so that a burst of changes makes one
  commit, and never more often than every 2 minutes, because the Hub limits commits per
  repository per hour. A failed upload is tried again, waiting twice as long each time.
- **When the Space is stopped**, the last changes are uploaded first. A crash can lose up
  to the last two minutes.
- Sessions are in the copy too, so a restart signs nobody out.
- The dataset must be private. It is checked before every download and upload, and a
  public one is refused. `slideseek deploy --accounts-repo` refuses a public one too.
- When an account is deleted, or an admin changes one, the dataset's history is squashed
  into one commit, so the account, or what it said before, does not live on in an old
  version. A copy small enough to be kept in git itself (the hosted demo's is) goes with
  the old commits; copies the Hub kept in its large-file storage are deleted from there
  too. The history is also squashed every 25 uploads to keep it short.

This puts the name, email address, phone number and password hash of everyone who signs
up in a private repository on your Hugging Face account. Anyone who can sign in to that
account, or who holds the token, can read them. That is why the token in step 3 is
limited to that one dataset.

### Other places

`SLIDESEEK_DATABASE_URL` names any database as a SQLAlchemy URL, and wins over
`SLIDESEEK_ACCOUNTS_REPO` when both are set. It holds a password, so on a Space it goes
under *Settings → Variables and secrets* as a **secret**, never as a variable: variables
are public. URLs that start with `postgres://` or `postgresql://`, as hosts hand them out,
are read as psycopg URLs.

- **Ports.** Hugging Face's forum, including replies from its staff, says a Space can only
  make outgoing connections on ports 80, 443 and 8080, and that other ports are opened
  only on request, for one Space at a time. PostgreSQL listens on 5432, so a hosted
  PostgreSQL may be unreachable from the Space. This was not tested here.
- **Persistent storage.** A Space with Hugging Face's paid persistent storage has a disk at
  `/data` that survives restarts, and the app uses `/data/slideseek-accounts.db` when it
  finds one.

With none of these, the Space logs a warning at start-up and keeps accounts in a file
that the next restart wipes.

### Admins

An admin opens *Manage accounts* in their profile, or goes to `/admin`, and can:

- see every account: name, username, email address, phone number, when it was made, and on
  how many devices it is signed in. Never a password, which is stored only as a hash;
- change an account's details, set a new password (which signs the account out
  everywhere), make it an admin or stop it being one, sign it out everywhere, or delete it;
- create accounts, admins included. An admin made this way can create admins too.

Every request is checked on the server, against the database, not against what the page
shows: an admin whose rights were removed is refused from their next request on. An admin
cannot remove their own admin rights, so the last admin cannot lock everyone out, and
deletes their own account from their profile, with its password, not from the list.
Admins' changes are logged by field name, never by value.

**The default admin** is the account whose username `SLIDESEEK_ADMIN` names
(`slideseek serve --admin`, `slideseek deploy --admin`). At start-up it is made an admin,
and from then on it is protected: no other admin can change it, sign it out or delete it,
it cannot stop being an admin, its username cannot be changed, and it cannot be deleted
from its profile either. Everything else about it, its owner can change.

Sign up first, and name the account afterwards. While no account has the username, nobody
can sign up with it, so that nobody else can claim it before its owner does; set too early,
it keeps the owner out as well. Then start once without the setting, sign up, and set it
again. A Space reads the variable when it starts, and changing a variable restarts it.

The sign-up form tells visitors that the people who run Slideseek can see their details.

## People's own lectures

With a library (`serve --library`, `SLIDESEEK_LIBRARIES`, or `deploy --libraries-repo`),
anyone signed in can add a recording from their own computer, search it beside the public
lectures, and play results from their own file. The recording is never uploaded. The work
is split where it is cheapest:

| Where | What | Code |
|---|---|---|
| The visitor's browser | 1 fps 160x90 grey frames, a still of each slide, the sound at 16 kHz, and speech recognition with Whisper `large-v3-turbo` (4-bit) on WebGPU through transformers.js | `static/capture/` |
| The server | Slide changes in the frames, OCR of the stills, moments and embeddings, with the indexer's own code and the searcher's models | `slideseek/library.py` |

- **The browser needs WebGPU**: Chrome or Edge on a computer. The speech model, 566 MB, is
  downloaded once from the Hugging Face Hub at a pinned revision and kept in the site's
  private files in the browser. On an Apple M4, an 84-minute lecture takes about 22 minutes
  ([measured](../docs/browser-indexing.md)), with the page open; closing it or a server
  restart loses at most the five minutes being transcribed.
- **What is sent**: the frames (about 14 MB compressed for 84 minutes), the stills (about
  4 MB) and the transcript. The server keeps each lecture's own small index and
  thumbnails, and deletes the full-size stills once it is built.
- **Private**: only the account that added a lecture can list, search or see it. An admin
  sees how many lectures each account has and how much space they take, never what they
  are. Deleting the account deletes them.
- **Limits** (`slideseek/library.py`): 20 lectures per account, 4 hours each, one being
  added at a time per account and 12 across the site; a lecture left unfinished for two
  days is removed.
- **One worker** reads the slides and embeds them for the whole site, a few at a time,
  under the same lock as searching, so a search waits a moment at most.

On a Space the disk is wiped at every restart, so each finished lecture is copied to the
private dataset `SLIDESEEK_LIBRARIES_REPO` (`slideseek/backup.py`, `LibraryBackup`): its
state and its own index, uploaded when it is ready, and restored at start-up. Deleting a
lecture or an account deletes its folder from the dataset, squashes the history, and deletes
the large files nothing points at any more. The dataset is checked to be private before
every upload, and `slideseek deploy` refuses a public one. A lecture still being added when
the Space restarts is not in the copy. Adding the same recording again starts it afresh,
but reuses what the browser had already transcribed, which it keeps until the server has it.
If the copy cannot be restored at start-up (most often, the token cannot reach the dataset),
the Space starts anyway with adding lectures switched off, and its log says why: searching
and accounts do not depend on it, and nothing is written to the dataset until it is fixed.

This puts the slide text, transcripts and thumbnails of the lectures people add in a
private repository on your Hugging Face account. Anyone who can sign in to that account, or
who holds the token, can read them.

## With Docker

```bash
docker build -t slideseek .
```

```bash
docker run -p 7860:7860 -v "$PWD/index:/home/user/index" slideseek
```

Any host that runs a Docker image works: set `SLIDESEEK_INDEX_REPO` (or mount an index at
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
| `SLIDESEEK_INDEX` | `index` | Directory holding the index. |
| `SLIDESEEK_INDEX_REPO` | unset | Dataset repository to download the index from when the directory is empty. |
| `SLIDESEEK_VIDEOS` | unset | Directory of local recordings to play in the page. Not read on a Space. |
| `SLIDESEEK_EXAMPLES` | `config/examples.json` | Example questions shown under the search box. |
| `SLIDESEEK_BENCHMARK` | `results/test/metrics.json` | Benchmark metrics to show; the page shows none if the file is absent. |
| `SLIDESEEK_DEVICE` | `cpu` | Device for the query encoders. Always `cpu` on ZeroGPU. |
| `SLIDESEEK_RERANK` | unset | `1` offers "Rerank on a GPU" in the Space's app. Not read by the container. |
| `SLIDESEEK_ACCOUNTS_REPO` | unset | Private dataset that keeps a copy of the accounts' SQLite file ([Accounts](#on-a-space-a-copy-in-a-private-dataset)). Ignored when `SLIDESEEK_DATABASE_URL` is set. |
| `SLIDESEEK_ACCOUNTS_TOKEN` | unset | Token that can write to that dataset; a Space **secret**. `HF_TOKEN` is used if this is unset. |
| `SLIDESEEK_ADMIN` | unset | Username of the default admin ([Admins](#admins)), an account that already exists. A variable, not a secret. |
| `SLIDESEEK_DATABASE_URL` | `sqlite:///accounts.db` | Where accounts are kept ([Accounts](#other-places)). On a Space, a secret; without it, `/data` if the Space has persistent storage. |
| `SLIDESEEK_LIBRARIES` | unset | Directory for the lectures people add themselves ([People's own lectures](#peoples-own-lectures)). Without it or the next, nobody can add any. |
| `SLIDESEEK_LIBRARIES_REPO` | unset | Private dataset that keeps a copy of those lectures; turns them on, in `libraries` (or `/data` with persistent storage) unless `SLIDESEEK_LIBRARIES` names a directory. |
| `SLIDESEEK_LIBRARIES_TOKEN` | unset | Token that can write to that dataset; a Space **secret**. The accounts' token is used if this is unset. |

## Why the frontend is not Gradio's

Gradio's components would have been the short way to a Space. Search needs no GPU (about
100 ms a query on a CPU, both encoders included), and a hand-written page gives full control
over the design, so the app is plain FastAPI with a static frontend, and Gradio supplies
only what ZeroGPU requires: the server and the queue in front of the one GPU call. When
reranking is offered, the page loads Gradio's JavaScript client, from this origin and only
once the switch is on, because that client is what tells ZeroGPU whose allowance a call
counts against.
