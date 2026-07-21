"""Statistiche per VIDEO (pool di tutti gli split), per decidere lo split per video."""
import os, glob, re
from collections import defaultdict

DATASETS = ["datasets/Pool Ball Detection V2.yolov8",
            "datasets/Pool Ball Detection V3.yolov8"]
SPLITS = ["train", "valid", "test"]
NAMES = ['0','1','10','11','12','13','14','15','2','3','4','5','6','7','8','9']

def vid_of(ds, fname):
    m = re.match(r'^([a-zA-Z]+)-(\d+)-(\d+)', fname)
    tag = "V2" if "V2" in ds else "V3"
    return f"{tag}/{m.group(1)}-{m.group(2)}" if m else f"{tag}/?"

imgs = defaultdict(int)
cls = defaultdict(lambda: defaultdict(int))
for ds in DATASETS:
    for s in SPLITS:
        for ip in glob.glob(os.path.join(ds, s, "images", "*.*")):
            name = os.path.basename(ip)
            v = vid_of(ds, name)
            imgs[v] += 1
            lp = os.path.join(ds, s, "labels", os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lp):
                for line in open(lp):
                    p = line.split()
                    if len(p) >= 5:
                        cls[v][int(float(p[0]))] += 1

print(f"{'video':10s} {'imgs':>5s}  " + " ".join(f"{NAMES[c]:>3s}" for c in range(16)))
for v in sorted(imgs):
    row = f"{v:10s} {imgs[v]:>5d}  " + " ".join(f"{cls[v][c]:>3d}" for c in range(16))
    print(row)
print(f"\n{'TOTALE':10s} {sum(imgs.values()):>5d}")
