"""Fetch the ULB credit-card fraud dataset (creditcard.csv, 150 MB) into data/.

    python scripts/get_data.py

The file is the one published by the Machine Learning Group of ULB on Kaggle (Database Contents
License). This script downloads TensorFlow's public copy of the same file, which needs no account,
and checks its SHA-256.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
TARGET = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"


def main() -> None:
    if TARGET.exists():
        print("creditcard.csv already present")
        return
    TARGET.parent.mkdir(exist_ok=True)
    blob = urllib.request.urlopen(URL, timeout=600).read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        sys.exit("checksum mismatch: %s" % digest)
    TARGET.write_bytes(blob)
    print("wrote", TARGET)


if __name__ == "__main__":
    main()
