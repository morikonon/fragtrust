"""Minimal read-only FAT32 parser: geometry, cluster chains and the directory tree (with long names).

Used to record block-level ground truth before the image is quick-formatted. Every file is
reassembled from its chain and compared with the source sha256, so a parsing error cannot pass
unnoticed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EOC = 0x0FFFFFF8
BAD = 0x0FFFFFF7
ATTR_LFN = 0x0F
ATTR_DIR = 0x10
ATTR_VOLUME = 0x08
DELETED = 0xE5


@dataclass(frozen=True)
class Geometry:
    bytes_per_sector: int
    sectors_per_cluster: int
    reserved_sectors: int
    n_fats: int
    fat_sectors: int
    total_sectors: int
    root_cluster: int

    @property
    def cluster_size(self) -> int:
        return self.bytes_per_sector * self.sectors_per_cluster

    @property
    def data_offset(self) -> int:
        return (self.reserved_sectors + self.n_fats * self.fat_sectors) * self.bytes_per_sector

    @property
    def n_clusters(self) -> int:
        data_sectors = self.total_sectors - self.reserved_sectors - self.n_fats * self.fat_sectors
        return data_sectors // self.sectors_per_cluster

    def cluster_offset(self, cluster: int) -> int:
        return self.data_offset + (cluster - 2) * self.cluster_size


@dataclass(frozen=True)
class FatFile:
    path: str
    size: int
    clusters: tuple[int, ...]

    @property
    def fragments(self) -> int:
        """Number of contiguous runs of clusters."""
        if not self.clusters:
            return 0
        return 1 + sum(b != a + 1 for a, b in zip(self.clusters[:-1], self.clusters[1:]))


def parse_boot_sector(sector: bytes) -> Geometry:
    if sector[510:512] != b"\x55\xaa":
        raise ValueError("missing boot sector signature 0x55AA")
    bps, spc, reserved, n_fats, root_entries, total16, _media, fat16 = struct.unpack_from("<HBHBHHBH", sector, 11)
    total32, fat32, _, _, root_cluster = struct.unpack_from("<IIHHI", sector, 32)
    if root_entries != 0 or fat16 != 0 or fat32 == 0:
        raise ValueError("not a FAT32 file system")
    return Geometry(bps, spc, reserved, n_fats, fat32, total32 or total16, root_cluster)


def _lfn_checksum(short_name: bytes) -> int:
    total = 0
    for byte in short_name:
        total = (((total & 1) << 7) + (total >> 1) + byte) & 0xFF
    return total


def _short_name(entry: bytes) -> str:
    base, ext, case = entry[0:8].rstrip(b" "), entry[8:11].rstrip(b" "), entry[12]
    if base[:1] == b"\x05":
        base = b"\xe5" + base[1:]
    base_s = base.decode("cp437")
    ext_s = ext.decode("cp437")
    if case & 0x08:
        base_s = base_s.lower()
    if case & 0x10:
        ext_s = ext_s.lower()
    return f"{base_s}.{ext_s}" if ext_s else base_s


def _lfn_part(entry: bytes) -> str:
    raw = entry[1:11] + entry[14:26] + entry[28:32]
    text = raw.decode("utf-16-le", errors="replace")
    return text.split("\x00", 1)[0].replace("￿", "")


class Fat32:
    def __init__(self, image: str | Path):
        self.image = Path(image)
        with open(self.image, "rb") as f:
            self.geometry = parse_boot_sector(f.read(512))
        g = self.geometry
        fat_offset = g.reserved_sectors * g.bytes_per_sector
        self.fat = np.fromfile(self.image, dtype="<u4", count=g.n_clusters + 2, offset=fat_offset) & 0x0FFFFFFF

    def chain(self, first: int) -> list[int]:
        if first < 2:
            return []
        clusters, seen, c = [], set(), first
        while 2 <= c < BAD:
            if c in seen or c >= self.geometry.n_clusters + 2:
                raise ValueError(f"corrupt cluster chain starting at {first}")
            seen.add(c)
            clusters.append(c)
            c = int(self.fat[c])
        return clusters

    def read_clusters(self, clusters: list[int] | tuple[int, ...]) -> bytes:
        size = self.geometry.cluster_size
        with open(self.image, "rb") as f:
            parts = []
            for c in clusters:
                f.seek(self.geometry.cluster_offset(c))
                parts.append(f.read(size))
        return b"".join(parts)

    def read_file(self, file: FatFile) -> bytes:
        return self.read_clusters(file.clusters)[: file.size]

    def free_clusters(self) -> int:
        return int((self.fat[2:] == 0).sum())

    def files(self) -> list[FatFile]:
        """All live regular files, depth-first, with '/'-separated paths."""
        out: list[FatFile] = []
        self._walk(self.geometry.root_cluster, "", out, set())
        return out

    def _walk(self, cluster: int, prefix: str, out: list[FatFile], visited: set[int]) -> None:
        if cluster in visited:
            return
        visited.add(cluster)
        data = self.read_clusters(self.chain(cluster))
        lfn: dict[int, str] = {}
        checksum = None
        for pos in range(0, len(data), 32):
            entry = data[pos : pos + 32]
            first, attr = entry[0], entry[11]
            if first == 0x00:
                break
            if first == DELETED:
                lfn, checksum = {}, None
                continue
            if attr == ATTR_LFN:
                order = first & 0x1F
                if first & 0x40:
                    lfn, checksum = {}, entry[13]
                lfn[order] = _lfn_part(entry)
                continue
            long_name = None
            if lfn and checksum == _lfn_checksum(entry[0:11]):
                long_name = "".join(lfn[k] for k in sorted(lfn))
            lfn, checksum = {}, None
            if attr & ATTR_VOLUME:
                continue
            name = long_name or _short_name(entry)
            if name in (".", ".."):
                continue
            hi, lo = struct.unpack_from("<H", entry, 20)[0], struct.unpack_from("<H", entry, 26)[0]
            start = (hi << 16) | lo
            path = f"{prefix}{name}"
            if attr & ATTR_DIR:
                self._walk(start, f"{path}/", out, visited)
            else:
                size = struct.unpack_from("<I", entry, 28)[0]
                out.append(FatFile(path, size, tuple(self.chain(start))))
