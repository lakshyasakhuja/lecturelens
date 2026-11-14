"""LectureLens: semantic search for lecture recordings.

The package is split along the line that defines the system:

* **Offline indexing** (GPU, run once): ``ingest`` -> ``slides`` -> ``ocr`` -> ``asr``
  -> ``moments`` -> ``embed`` -> ``index``.
* **Online search** (CPU): ``index`` -> ``search``.

Heavy dependencies (torch, faster-whisper, the OCR runtime) are imported lazily inside the
functions that need them, so importing this package, running the CLI's ``--help`` and running
the unit tests never loads a model.
"""

__version__ = "0.1.0"
