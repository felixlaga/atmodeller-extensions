"""Downloads the authors' files of Hakim et al. (2026) from OSF and checks their SHA-256 hashes

The files are not stored in this repository. They are the output workbooks for the fully
molten TOI-421 b MEB at 1x and 100x solar metallicity and the notebook that made them, from
https://osf.io/ac6bw (Data Availability of arXiv:2508.19235).

    uv run python research/condensed_volumes/fetch_sources.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request

from common import SOURCES

FILES: dict[str, tuple[str, str]] = {
    "HHeCNOSi_magma_sol_real_1xsolar_melt100.xlsx": (
        "https://osf.io/download/69315a3c95a56f20be39be52/",
        "c0627278c4d729950654e27f5dae9fb7ca1a27c9b96e2e6002635a1c1540f1eb",
    ),
    "HHeCNOSi_magma_sol_real_100xsolar_melt100.xlsx": (
        "https://osf.io/download/69315a3cbbc8f5ed0e39bdab/",
        "496ddc27a500f185877b52e6e4f60f75efaf1104f3a6bfe005c42caa65c3f04f",
    ),
    "generate_data.ipynb": (
        "https://osf.io/download/2sh5w/",
        "fd24f19878407ca5b04b082ca0d0bec86e94c2035be561f9c0b1179b6a8929f6",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    SOURCES.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, (url, expected) in FILES.items():
        path = SOURCES / name
        if path.exists() and sha256(path.read_bytes()) == expected:
            print(f"{name}: present, hash matches")
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if sha256(data) != expected:
                raise RuntimeError(f"{name}: hash mismatch, got {sha256(data)}")
            path.write_bytes(data)
            print(f"{name}: downloaded, hash matches")
        record[name] = {"url": url, "sha256": expected}
    (SOURCES / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
