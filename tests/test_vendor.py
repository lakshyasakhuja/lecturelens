"""Fetching the browser's speech-recognition scripts at pinned versions."""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile

import pytest

from slideseek.web import vendor
from slideseek.web.vendor import Package, VendorError


def tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def integrity(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()


def pinned(monkeypatch, data: bytes, files: dict[str, bytes], *, record=True, check=None):
    """Pin one package whose tarball is ``data``, holding ``files``."""
    package = Package(
        name="@example/pkg",
        version="1.2.3",
        integrity=check or integrity(data),
        files={
            member: (member.rsplit("/", 1)[-1], hashlib.sha256(body).hexdigest() if record else "")
            for member, body in files.items()
        },
    )
    monkeypatch.setattr(vendor, "PACKAGES", (package,))
    return package


def test_fetches_the_pinned_files_and_then_nothing(tmp_path, monkeypatch):
    files = {"package/dist/a.js": b"export const a = 1;", "package/dist/b.wasm": b"\0asm"}
    data = tarball(files)
    package = pinned(monkeypatch, data, files)
    asked = []

    def download(url):
        asked.append(url)
        return data

    assert vendor.missing(tmp_path) == ["a.js", "b.wasm"]
    assert sorted(vendor.fetch(tmp_path, download=download)) == ["a.js", "b.wasm"]
    assert asked == [package.tarball]
    assert package.tarball == "https://registry.npmjs.org/@example/pkg/-/pkg-1.2.3.tgz"
    assert (tmp_path / "a.js").read_bytes() == files["package/dist/a.js"]
    assert vendor.missing(tmp_path) == []
    assert vendor.fetch(tmp_path, download=download) == []
    assert len(asked) == 1


def test_a_changed_file_is_fetched_again(tmp_path, monkeypatch):
    files = {"package/dist/a.js": b"export const a = 1;"}
    data = tarball(files)
    pinned(monkeypatch, data, files)
    vendor.fetch(tmp_path, download=lambda url: data)
    (tmp_path / "a.js").write_text("tampered")
    assert vendor.missing(tmp_path) == ["a.js"]
    assert vendor.fetch(tmp_path, download=lambda url: data) == ["a.js"]
    assert (tmp_path / "a.js").read_bytes() == files["package/dist/a.js"]


def test_a_package_that_does_not_match_the_registry_is_refused(tmp_path, monkeypatch):
    files = {"package/dist/a.js": b"export const a = 1;"}
    data = tarball(files)
    pinned(monkeypatch, data, files, check=integrity(b"something else"))
    with pytest.raises(VendorError, match="checksum"):
        vendor.fetch(tmp_path, download=lambda url: data)
    assert not (tmp_path / "a.js").exists()


def test_a_file_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    # The registry vouches for the tarball, but the file in it is not the one recorded.
    swapped = tarball({"package/dist/a.js": b"a different file"})
    pinned(monkeypatch, swapped, {"package/dist/a.js": b"the file that was tested"})
    with pytest.raises(VendorError, match="not the file that was pinned"):
        vendor.fetch(tmp_path, download=lambda url: swapped)
    assert list(tmp_path.iterdir()) == []


def test_every_pinned_file_has_its_hash_recorded():
    for package in vendor.PACKAGES:
        assert package.integrity.startswith("sha512-")
        for name, digest in package.files.values():
            assert len(digest) == 64, name
