"""Step 3 (command line): pretrain on ODIR after prep_odir.py has run.
   python pretrain_odir.py --arm binary --odir-out odir_out --images odir_pack/images [--quick]"""
import argparse
from pathlib import Path
from src.pretrain import pretrain


def main():


    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["binary", "multi"], required=True)
    ap.add_argument("--odir-out", required=True, help="folder with odir_manifest.csv (written by prep_odir.py)")
    ap.add_argument("--images", required=True, help="folder with the preprocessed 256-px ODIR PNGs")
    ap.add_argument("--seed", type=int, default=42, help="PRETRAIN_SEED (default 42)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    pretrain(a.arm, Path(a.odir_out) / "odir_manifest.csv", a.images, a.odir_out, quick=a.quick, seed=a.seed)


if __name__ == "__main__":  # guard needed for multiprocessing on Mac/Windows
    main()
