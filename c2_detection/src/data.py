"""Dataset helpers: find the CSV files and work out each file's class label.

CICIoMT2024 CSVs have NO label column. The label comes from the file name:
    'TCP_IP-DDoS-SYN3_train.pcap.csv'  ->  'TCP_IP-DDoS-SYN'
"""
import re
from pathlib import Path

from config import CLASSES

# matches the ending '_train.pcap.csv' or '_test.pcap.csv'
_SUFFIX = re.compile(r"_(train|test)\.pcap\.csv$", re.IGNORECASE)


def label_from_filename(name: str) -> str:
    """'TCP_IP-DDoS-SYN3_train.pcap.csv' -> 'TCP_IP-DDoS-SYN'"""
    base = _SUFFIX.sub("", name)       # remove '_train.pcap.csv'
    base = re.sub(r"\d+$", "", base)   # remove the part number at the end (1, 2, 3...)
    return base


def list_csvs(split_dir: Path) -> dict[str, list[Path]]:
    """Group the CSV files of one folder (Train or Test) by class.
    Returns {class_name: [file paths]} in classes.json order."""
    by_class = {c: [] for c in CLASSES}
    for path in sorted(split_dir.glob("*.csv")):
        label = label_from_filename(path.name)
        if label in by_class:
            by_class[label].append(path)
        else:
            print(f"[warn] unknown class '{label}' from file {path.name} - skipped")

    missing = [c for c, files in by_class.items() if not files]
    if missing:
        print(f"[warn] no files for: {missing}")
    return by_class