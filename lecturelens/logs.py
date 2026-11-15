"""Logging setup shared by the CLI and the evaluation scripts."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

# Libraries that narrate every HTTP request and model load at INFO.
_NOISY = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "faster_whisper",
    "RapidOCR",
    "pyscenedetect",
    "torch.distributed",
)


def configure_logging(console: Console | None = None, verbose: bool = False) -> None:
    """Show LectureLens's own progress; keep third-party libraries quiet unless asked.

    Call this before anything imports transformers or huggingface_hub: several of the
    switches are environment variables those libraries read once, at import.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console or Console(), show_path=False, show_time=False, markup=False
            )
        ],
        force=True,
    )
    logging.getLogger("lecturelens").setLevel(logging.DEBUG if verbose else logging.INFO)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.ERROR)
    # torch logs a note about stream redirection on every import on macOS, through its own
    # handler, which resets levels; disabling the logger is the only thing that sticks.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").disabled = True
