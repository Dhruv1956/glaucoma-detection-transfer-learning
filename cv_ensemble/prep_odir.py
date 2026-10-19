"""Steps 2 (command line): build per-eye ODIR labels, preprocess, de-duplicate vs the project images.
   python prep_odir.py --odir /kaggle/input/ocular-disease-recognition-odir5k --pack pack --out odir_out --work odir_pack
Exit code 2 = an ODIR image matched a project TEST image (audit is written; do not continue)."""
import argparse
import sys
from src.odir import prepare_odir


def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--odir", required=True)
    ap.add_argument("--targets", default="target_hashes.csv")
    ap.add_argument("--pack", required=True, help="the 256-px pack/ folder (used only for train/val pixel checks)")
    ap.add_argument("--out", default="odir_out")
    ap.add_argument("--work", default="odir_pack")
    ap.add_argument("--arm", choices=["binary", "multi"], default=None, help="preprocess only this arm (default: both)")
    ap.add_argument("--quick-n", type=int, default=None)
    a = ap.parse_args()
    res = prepare_odir(a.odir, a.targets, a.pack, a.out, a.work, quick_n=a.quick_n, arm=a.arm)
    sys.exit(2 if res["test_matches"] else 0)


if __name__ == "__main__":  # guard needed for multiprocessing on Mac/Windows
    main()
