"""
Costruisce il dataset UNITO (V2+V3) a 16 classi, con SPLIT PER VIDEO.

- Metto insieme tutti i frame dei due dataset (ignorando gli split originali).
- Assegno ogni VIDEO intero a train o valid (mai spezzato) -> niente leakage.
- Scrivo un unico data.yaml a 16 classi.

Assegnazione (vedi analyze_videos.py):
  VALID = V3/video-3  (dall'alto, mai visto in training)
  TRAIN = tutti gli altri 5 video
Nota: le classi 11,12,13,15 non sono in valid (non esistono in video-3);
restano in train dove servono. Giudice finale = il nostro video.
"""
import os, glob, re, shutil
from collections import defaultdict

SRC = ["datasets/Pool Ball Detection V2.yolov8",
       "datasets/Pool Ball Detection V3.yolov8"]
SPLITS = ["train", "valid", "test"]
NAMES = ['0','1','10','11','12','13','14','15','2','3','4','5','6','7','8','9']
OUT = "datasets/merged"
VALID_VIDEOS = {"V3/video-3"}


def vid_of(ds, fname):
    m = re.match(r'^([a-zA-Z]+)-(\d+)-(\d+)', fname)
    tag = "V2" if "V2" in ds else "V3"
    return f"{tag}/{m.group(1)}-{m.group(2)}" if m else f"{tag}/?"


def main():
    # cartelle di destinazione pulite
    for split in ["train", "valid"]:
        for sub in ["images", "labels"]:
            d = os.path.join(OUT, split, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    counts = defaultdict(int)
    cls_counts = defaultdict(lambda: defaultdict(int))
    for ds in SRC:
        for s in SPLITS:
            for ip in glob.glob(os.path.join(ds, s, "images", "*.*")):
                name = os.path.basename(ip)
                stem = os.path.splitext(name)[0]
                lp = os.path.join(ds, s, "labels", stem + ".txt")
                v = vid_of(ds, name)
                dst = "valid" if v in VALID_VIDEOS else "train"
                shutil.copy2(ip, os.path.join(OUT, dst, "images", name))
                if os.path.exists(lp):
                    shutil.copy2(lp, os.path.join(OUT, dst, "labels", stem + ".txt"))
                    for line in open(lp):
                        p = line.split()
                        if len(p) >= 5:
                            cls_counts[dst][int(float(p[0]))] += 1
                counts[dst] += 1

    # data.yaml
    abspath = os.path.abspath(OUT).replace("\\", "/")
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {abspath}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write(f"nc: {len(NAMES)}\n")
        f.write("names: [" + ", ".join(f"'{n}'" for n in NAMES) + "]\n")

    print(f"Costruito in {OUT}/")
    print(f"  train: {counts['train']} img   valid: {counts['valid']} img")
    print("  classi in TRAIN:", {NAMES[c]: cls_counts['train'][c] for c in range(16)})
    print("  classi in VALID:", {NAMES[c]: cls_counts['valid'][c] for c in range(16)})
    missing = [NAMES[c] for c in range(16) if cls_counts['valid'][c] == 0]
    print("  classi ASSENTI in valid:", missing)


if __name__ == "__main__":
    main()
