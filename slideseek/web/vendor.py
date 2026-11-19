"""The scripts the page runs speech recognition with, fetched at pinned versions.

Adding a lecture from the browser runs Whisper on the visitor's own GPU, through
transformers.js and the WebAssembly build of ONNX Runtime it drives. The page loads scripts
from its own origin only, so these are served from ``static/vendor/browser/`` like the
rest of the page. They are not kept in git: the runtime alone is 27 MB, and a new version
would add as much again to the history every time. Instead, each file is fetched from the
npm registry at the version named here, the package is checked against the registry's own
SHA-512, and the file against a SHA-256 recorded here, so what is served is exactly what
was tested.

    python -m slideseek vendor        fetch whatever is missing or does not match

Licences: transformers.js is Apache-2.0, ONNX Runtime is MIT, as their packages state.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
VENDOR_DIR = STATIC_DIR / "vendor" / "browser"
REGISTRY = "https://registry.npmjs.org"


@dataclass(frozen=True)
class Package:
    name: str
    version: str
    integrity: str  # the registry's "sha512-..." for the tarball
    files: dict[str, tuple[str, str]]  # path in the package -> (file name served, SHA-256)

    @property
    def tarball(self) -> str:
        base = self.name.rsplit("/", 1)[-1]
        return f"{REGISTRY}/{self.name}/-/{base}-{self.version}.tgz"


PACKAGES = (
    Package(
        name="@huggingface/transformers",
        version="4.3.2",
        integrity="sha512-V3+dqf3iWlRYJ9I9GreM3zGQd0lAXnnoPeddmkOQYWAEE4l/+Fi6p7tx/qVUJlKQgLXGim2W76rTOzz+eC0i7g==",
        files={
            "package/dist/transformers.min.js": (
                "transformers.min.js",
                "8b5a3171f8876bab2b1820a7d65eb87628e3d30bb5ea85b9dd398e76eb34decb",
            ),
        },
    ),
    # The ONNX Runtime build transformers.js 4.3.2 was released against (its package.json).
    Package(
        name="onnxruntime-web",
        version="1.31.0-dev.20260914-8d85527a0",
        integrity="sha512-Iy7rtadoBgxS/LLvDr3QW38DB1PNXRnr0GJMcL0TAt7c9qjgVQl83UlGCVyeAnK2InpmW8Uc3PL8XIuqtDeF6g==",
        files={
            "package/dist/ort-wasm-simd-threaded.asyncify.mjs": (
                "ort-wasm-simd-threaded.asyncify.mjs",
                "0966b6105cd936744498aa60df7a22cbd47af3374dbc64a9ab561c08a71e3611",
            ),
            "package/dist/ort-wasm-simd-threaded.asyncify.wasm": (
                "ort-wasm-simd-threaded.asyncify.wasm",
                "49871f5a4409519797e127440868a6d1923339d9185907f301a5b2a1d90af082",
            ),
        },
    ),
)


class VendorError(RuntimeError):
    """A file could not be fetched, or is not the file that was pinned."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def missing(dest: Path = VENDOR_DIR) -> list[str]:
    """The served files that are absent, or differ from the pinned ones."""
    wrong = []
    for package in PACKAGES:
        for name, expected in package.files.values():
            path = dest / name
            if not path.is_file() or (expected and sha256(path) != expected):
                wrong.append(name)
    return wrong


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "slideseek-vendor"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _check_integrity(data: bytes, integrity: str, what: str) -> None:
    algorithm, _, expected = integrity.partition("-")
    if algorithm != "sha512":
        raise VendorError(f"{what}: unsupported integrity {algorithm}")
    actual = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if actual != expected:
        raise VendorError(f"{what} does not match the registry's checksum; not using it")


def fetch(dest: Path = VENDOR_DIR, *, download=_download) -> list[str]:
    """Fetch every file that is missing or wrong. Returns the names of the files written."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(missing(dest))
    written = []
    for package in PACKAGES:
        names = {member: served for member, served in package.files.items() if served[0] in wanted}
        if not names:
            continue
        what = f"{package.name}@{package.version}"
        log.info("fetching %s", what)
        data = download(package.tarball)
        _check_integrity(data, package.integrity, what)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member, (name, expected) in names.items():
                entry = archive.getmember(member)
                if not entry.isfile():
                    raise VendorError(f"{what}: {member} is not a file")
                extracted = archive.extractfile(entry)
                if extracted is None:
                    raise VendorError(f"{what}: {member} cannot be read")
                tmp = dest / f".{name}.partial"
                tmp.write_bytes(extracted.read())
                if expected and sha256(tmp) != expected:
                    tmp.unlink()
                    raise VendorError(f"{what}: {member} is not the file that was pinned")
                os.replace(tmp, dest / name)
                written.append(name)
    return written
