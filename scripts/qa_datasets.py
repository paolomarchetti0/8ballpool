"""
QA dei due dataset PRIMA del training YOLO.

Controlli (nessun training):
  1. Conteggi immagini/label per split.
  2. BILANCIAMENTO CLASSI: istanze per classe (0-15) per dataset e combinato.
  3. DUPLICATI / QUASI-DUPLICATI via average-hash (aHash 8x8):
     - cluster di quasi-duplicati dentro ogni dataset,
     - LEAKAGE: immagini quasi uguali tra split diversi (train vs valid/test),
     - sovrapposizione tra V2 e V3.

L'identita' della classe: nel data.yaml l'ordine names e' string-sort
['0','1','10','11','12','13','14','15','2','3','4','5','6','7','8','9'],
quindi l'indice YOLO i -> etichetta names[i] (es. indice 2 = pallina '10').
"""
import os
import re
import glob
import cv2
import numpy as np
from collections import defaultdict

DATASETS = [
    "datasets/Pool Ball Detection V2.yolov8",
    "datasets/Pool Ball Detection V3.yolov8",
]
SPLITS = ["train", "valid", "test"]
NAMES = ['0', '1', '10', '11', '12', '13', '14', '15',
         '2', '3', '4', '5', '6', '7', '8', '9']
HAMMING_DUP = 5    # distanza di Hamming <= questa -> quasi-duplicati


# ---------- utilita' ----------
def label_files(ds, split):
    return glob.glob(os.path.join(ds, split, "labels", "*.txt"))

def image_files(ds, split):
    d = os.path.join(ds, split, "images")
    return glob.glob(os.path.join(d, "*.*"))

def phash(path):
    """perceptual hash (DCT) 8x8 dalle basse frequenze -> intero a 64 bit.
    Piu' discriminante dell'aHash su immagini con grandi zone uniformi."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)
    low = dct[:8, :8]                 # basse frequenze (struttura grossa)
    med = np.median(low[1:, 1:])      # escludo il DC (low[0,0])
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def hamming(a, b):
    return bin(a ^ b).count("1")

def parse_name(fname):
    """'out-1-101_png.rf.xxx.png' -> (video='out-1', frame=101). None se non combacia."""
    m = re.match(r'^([a-zA-Z]+)-(\d+)-(\d+)', fname)
    if not m:
        return None, None
    return f"{m.group(1)}-{m.group(2)}", int(m.group(3))


# ---------- 1-2. conteggi e bilanciamento ----------
def class_counts(ds):
    per_split = {s: defaultdict(int) for s in SPLITS}
    n_imgs = {}
    for s in SPLITS:
        n_imgs[s] = len(image_files(ds, s))
        for lf in label_files(ds, s):
            with open(lf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(float(line.split()[0]))
                    per_split[s][cid] += 1
    return n_imgs, per_split


def report_counts():
    combined = defaultdict(int)
    for ds in DATASETS:
        print("=" * 70)
        print(os.path.basename(ds))
        n_imgs, per_split = class_counts(ds)
        tot_imgs = sum(n_imgs.values())
        print(f"  immagini: train={n_imgs['train']} valid={n_imgs['valid']} "
              f"test={n_imgs['test']}  (tot {tot_imgs})")
        ds_tot = defaultdict(int)
        for s in SPLITS:
            for cid, c in per_split[s].items():
                ds_tot[cid] += c
                combined[cid] += c
        print("  istanze per classe (etichetta: totale):")
        line = "   "
        for cid in range(16):
            line += f" {NAMES[cid]:>3}={ds_tot[cid]:<4}"
            if cid == 7:
                line += "\n   "
        print(line)
        vals = [ds_tot[c] for c in range(16)]
        print(f"  -> min={min(vals)} max={max(vals)} "
              f"squilibrio max/min={max(vals)/max(1,min(vals)):.1f}x")
    print("=" * 70)
    print("COMBINATO (V2+V3), istanze per classe:")
    line = "   "
    for cid in range(16):
        line += f" {NAMES[cid]:>3}={combined[cid]:<4}"
        if cid == 7:
            line += "\n   "
    print(line)
    vals = [combined[c] for c in range(16)]
    print(f"  -> totale istanze={sum(vals)}  min={min(vals)} max={max(vals)} "
          f"squilibrio={max(vals)/max(1,min(vals)):.1f}x")


# ---------- 3a. duplicati percettivi (pHash) ----------
def report_phash_duplicates():
    print("=" * 70)
    print("QUASI-DUPLICATI percettivi (pHash DCT, Hamming <= %d)" % HAMMING_DUP)
    records = []  # (hash, ds, split, name)
    for ds in DATASETS:
        for s in SPLITS:
            for p in image_files(ds, s):
                h = phash(p)
                if h is not None:
                    records.append((h, os.path.basename(ds)[-14:-8], s, os.path.basename(p)))
    print(f"  immagini processate: {len(records)}")

    n = len(records)
    hs = [r[0] for r in records]
    exact = near = leak = cross = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming(hs[i], hs[j])
            if d == 0:
                exact += 1
            if d <= HAMMING_DUP:
                near += 1
                if records[i][1] != records[j][1]:
                    cross += 1
                elif records[i][2] != records[j][2]:
                    leak += 1
    print(f"  coppie con pHash IDENTICO (d=0): {exact}")
    print(f"  coppie quasi-duplicate (d<=%d):  {near}" % HAMMING_DUP)
    print(f"    di cui leakage tra split:      {leak}")
    print(f"    di cui tra dataset V2 vs V3:   {cross}")


# ---------- 3b. leakage per frame consecutivi (dai nomi file) ----------
def report_frame_leakage():
    print("=" * 70)
    print("LEAKAGE per FRAME CONSECUTIVI (dai nomi file video-frame)")
    for ds in DATASETS:
        # video -> lista (frame, split)
        vids = defaultdict(list)
        unparsed = 0
        for s in SPLITS:
            for p in image_files(ds, s):
                vid, fr = parse_name(os.path.basename(p))
                if vid is None:
                    unparsed += 1
                else:
                    vids[vid].append((fr, s))
        # per ogni video, ordino per frame e conto vicini in split diversi
        leak_adjacent = 0   # frame consecutivi (diff piccola) in split diversi
        total_adjacent = 0
        gap = 3             # frame considerati "vicini" se distano <= gap
        for vid, lst in vids.items():
            lst.sort()
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    if lst[b][0] - lst[a][0] > gap:
                        break
                    total_adjacent += 1
                    if lst[a][1] != lst[b][1]:
                        leak_adjacent += 1
        pct = 100 * leak_adjacent / max(1, total_adjacent)
        print(f"  {os.path.basename(ds)}")
        print(f"    video distinti: {len(vids)}  (nomi non interpretati: {unparsed})")
        print(f"    coppie di frame vicini (<= {gap}): {total_adjacent}, "
              f"di cui in SPLIT DIVERSI: {leak_adjacent}  ({pct:.0f}%)")


if __name__ == "__main__":
    report_counts()
    report_phash_duplicates()
    report_frame_leakage()
