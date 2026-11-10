"""Split the official DAVIS pickle into independently readable research records."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/share/yty/heft/data"))
    args = parser.parse_args()
    source = args.root / "tapvid_davis/tapvid_davis.pkl"
    with source.open("rb") as f:
        records = pickle.load(f)
    out = args.root / "davis_records"
    out.mkdir(exist_ok=True)
    manifest = []
    for name in sorted(records):
        r = records[name]
        np.save(out / f"{name}.video.npy", np.asarray(r["video"]))
        np.savez_compressed(out / f"{name}.annotations.npz", points=r["points"], occluded=r["occluded"])
        # Hash-based partition fixed before reading any tracking scores.
        split = "calibration" if int(hashlib.sha256(name.encode()).hexdigest(), 16) % 3 == 0 else "evaluation"
        manifest.append({"name": name, "frames": len(r["video"]), "points": len(r["points"]), "split": split})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
