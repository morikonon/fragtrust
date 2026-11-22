"""Header-signature baseline (what a classic carver sees).

A signature matches only at the start of a cluster, where files begin on FAT. Several formats
share a container (ZIP, OLE2, TIFF, ISO-BMFF), so a match yields a set of candidate types.
Text formats have no signature and are never claimed. Entries are checked in order; the first
match wins, so specific signatures precede generic ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Part = tuple[int, bytes]

TIFF_FAMILY = ("tiff", "dng", "nef", "nrw", "arw", "pef", "3fr", "gpr")
ZIP_FAMILY = ("zip", "docx", "xlsx", "pptx", "jar", "apk", "key")


@dataclass(frozen=True)
class Signature:
    parts: tuple[Part, ...]
    classes: tuple[str, ...]


def _sig(classes: str | tuple[str, ...], *parts: Part) -> Signature:
    return Signature(tuple(parts), (classes,) if isinstance(classes, str) else classes)


def _alternatives(classes: str | tuple[str, ...], offset: int, *magics: bytes) -> list[Signature]:
    return [_sig(classes, (offset, m)) for m in magics]


SIGNATURES: list[Signature] = [
    _sig("jpg", (0, b"\xff\xd8\xff")),
    _sig("png", (0, b"\x89PNG\r\n\x1a\n")),
    *_alternatives("gif", 0, b"GIF87a", b"GIF89a"),
    _sig("psd", (0, b"8BPS")),
    _sig("cr2", (0, b"II*\x00"), (8, b"CR")),
    *_alternatives("orf", 0, b"IIRO", b"IIRS", b"MMOR"),
    _sig("rw2", (0, b"IIU\x00")),
    _sig("raf", (0, b"FUJIFILMCCD-RAW")),
    *_alternatives(TIFF_FAMILY, 0, b"II*\x00", b"MM\x00*"),
    *[_sig("heic", (4, b"ftyp"), (8, brand)) for brand in (b"heic", b"heix", b"mif1", b"msf1")],
    _sig("mov", (4, b"ftyp"), (8, b"qt  ")),
    _sig("m4a", (4, b"ftyp"), (8, b"M4A ")),
    _sig("3gp", (4, b"ftyp"), (8, b"3gp")),
    _sig(("mp4", "mov", "m4a", "3gp", "heic"), (4, b"ftyp")),
    _sig("avi", (0, b"RIFF"), (8, b"AVI ")),
    _sig("wav", (0, b"RIFF"), (8, b"WAVE")),
    *[_sig("aiff", (0, b"FORM"), (8, kind)) for kind in (b"AIFF", b"AIFC")],
    _sig("djvu", (0, b"AT&TFORM")),
    _sig(("mkv", "webm"), (0, b"\x1a\x45\xdf\xa3")),
    _sig(("ogg", "ogv"), (0, b"OggS")),
    _sig("flac", (0, b"fLaC")),
    _sig("mp3", (0, b"ID3")),
    _sig("wma", (0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11")),
    _sig(("pdf", "ai"), (0, b"%PDF-")),
    _sig(("eps", "ai"), (0, b"%!PS-Adobe")),
    _sig("eps", (0, b"\xc5\xd0\xd3\xc6")),
    _sig("epub", (0, b"PK\x03\x04"), (30, b"mimetypeapplication/epub+zip")),
    _sig(ZIP_FAMILY, (0, b"PK\x03\x04")),
    _sig(("doc", "xls", "ppt", "msi"), (0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")),
    _sig("rtf", (0, b"{\\rtf")),
    _sig("7z", (0, b"7z\xbc\xaf\x27\x1c")),
    _sig("rar", (0, b"Rar!\x1a\x07")),
    _sig("gz", (0, b"\x1f\x8b\x08")),
    _sig("bz2", (0, b"BZh")),
    _sig("xz", (0, b"\xfd7zXZ\x00")),
    _sig("deb", (0, b"!<arch>\ndebian")),
    _sig("rpm", (0, b"\xed\xab\xee\xdb")),
    _sig("pkg", (0, b"xar!")),
    _sig("elf", (0, b"\x7fELF")),
    *_alternatives("mach-o", 0, b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"),
    _sig(("exe", "dll"), (0, b"MZ")),
    _sig("sqlite", (0, b"SQLite format 3\x00")),
    *_alternatives("pcap", 0, b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a"),
    *_alternatives("ttf", 0, b"\x00\x01\x00\x00\x00", b"OTTO", b"true"),
    _sig("dwg", (0, b"AC10")),
    _sig("mobi", (60, b"BOOKMOBI")),
    _sig("xml", (0, b"<?xml")),
    *_alternatives("html", 0, b"<!DOCTYPE html", b"<!doctype html", b"<html", b"<HTML"),
    _sig("tex", (0, b"\\documentclass")),
]


def match_headers(blocks: np.ndarray) -> np.ndarray:
    """Index into SIGNATURES of the first signature matching each block, -1 if none."""
    out = np.full(len(blocks), -1, dtype=np.int32)
    for i, sig in enumerate(SIGNATURES):
        hit = out == -1
        for offset, magic in sig.parts:
            hit &= (blocks[:, offset : offset + len(magic)] == np.frombuffer(magic, dtype=np.uint8)).all(axis=1)
        out[hit] = i
    return out


def propagate(headers: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """Contiguous-carving assumption: a header's type extends until the next header or an empty block."""
    out = np.empty_like(headers)
    current = -1
    for i, (h, e) in enumerate(zip(headers, empty)):
        current = h if h >= 0 else (-1 if e else current)
        out[i] = current
    return out


def candidate_sets(signature_ids: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Boolean (n, n_classes) candidate matrix; an all-false row means no answer."""
    index = {name: i for i, name in enumerate(class_names)}
    table = np.zeros((len(SIGNATURES) + 1, len(class_names)), dtype=bool)
    for i, sig in enumerate(SIGNATURES):
        table[i, [index[c] for c in sig.classes if c in index]] = True
    return table[signature_ids]
