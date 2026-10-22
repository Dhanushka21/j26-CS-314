"""STEP 4 check: build windows and prove they are correct.

Run from repo root:   python c2_detection/tools/check_windows.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C                                                         # noqa: E402
from src.data import list_csvs, read_features, window_starts, build_split  # noqa: E402


def main():
    # 1. small example of window_starts
    print("window_starts(25, 10, 5) =", window_starts(25, 10, 5), " (expected [ 0  5 10 15])")

    # 2. a window must be exactly 10 consecutive rows of the file
    f = list_csvs(C.TEST_DIR)["Recon-Ping_Sweep"][0]
    x = read_features(f)
    s = window_starts(len(x), C.WINDOW_SIZE, C.TEST_STRIDE)
    print(f"\n{f.name}: rows={len(x)}  ->  windows={len(s)}")
    w = x[s[3]:s[3] + C.WINDOW_SIZE]
    print("window #3 == rows 30..39 of the file:", np.array_equal(w, x[30:40]))

    # 3. build the real test windows
    print("\n== building TEST windows (stride", C.TEST_STRIDE, ", cap", C.MAX_WINDOWS_PER_CLASS_TEST, ")")
    t0 = time.time()
    X, y = build_split(C.TEST_DIR, C.WINDOW_SIZE, C.TEST_STRIDE, C.MAX_WINDOWS_PER_CLASS_TEST, C.SEED)
    print(f"\nX shape = {X.shape}   y shape = {y.shape}   ({time.time() - t0:.0f}s)")
    print(f"memory  = {X.nbytes / 1e6:.0f} MB")
    print("classes present:", len(np.unique(y)), "/ 19")


if __name__ == "__main__":
    main()