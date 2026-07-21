"""
QA qualita' label: disegno i bounding box YOLO + etichetta classe su alcune
immagini di V2 e V3, per verificare a occhio che le annotazioni siano giuste
(box ben posizionati, numero pallina corretto).

Label YOLO: 'class cx cy w h' normalizzati in [0,1].
Etichetta mostrata = NAMES[class] (indice string-sort, vedi qa_datasets.py).
"""
import os
import glob
import cv2

DATASETS = [
    "datasets/Pool Ball Detection V2.yolov8",
    "datasets/Pool Ball Detection V3.yolov8",
]
NAMES = ['0', '1', '10', '11', '12', '13', '14', '15',
         '2', '3', '4', '5', '6', '7', '8', '9']
OUT_DIR = "output"
N_PER_DS = 6      # immagini campione per dataset


def draw_labels(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                cid = int(float(p[0]))
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, NAMES[cid], (x1, max(0, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in DATASETS:
        imgs = sorted(glob.glob(os.path.join(ds, "train", "images", "*.*")))
        # campiono spalmato sull'elenco
        step = max(1, len(imgs) // N_PER_DS)
        sample = imgs[::step][:N_PER_DS]
        tiles = []
        for ip in sample:
            name = os.path.basename(ip)
            lp = os.path.join(ds, "train", "labels",
                              os.path.splitext(name)[0] + ".txt")
            vis = draw_labels(ip, lp)
            if vis is not None:
                vis = cv2.resize(vis, (480, 270))
                tiles.append(vis)
        # griglia 2 x 3
        rows = [cv2.hconcat(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        grid = cv2.vconcat(rows)
        tag = "V2" if "V2" in ds else "V3"
        out = os.path.join(OUT_DIR, f"labels_{tag}.png")
        cv2.imwrite(out, grid)
        print(f"{tag}: salvato {out}  ({len(tiles)} immagini)")


if __name__ == "__main__":
    main()
