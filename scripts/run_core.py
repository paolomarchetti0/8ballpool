"""
Driver del NUCLEO su un frame: tavolo + buche + palle (identita' YOLO native).

Uso:
  python scripts/run_core.py "datasets/videos/video2.mp4" 69

Riusa le funzioni gia' scritte (detect_table, detect_pockets) e disegna tutto
su un overlay unico. Le palle qui usano l'ETICHETTA YOLO (il video2 e' nel
dominio del dataset, quindi i numeri YOLO sono affidabili).
"""
import sys
import cv2
import numpy as np
from ultralytics import YOLO

from detect_table import cloth_mask, largest_contour, order_corners, TOP_W, TOP_H
from detect_pockets import find_pockets
from detect_balls import ball_features, label_from_features

WEIGHTS = "runs/detect/full/weights/best.pt"
NAMES = ['0', '1', '10', '11', '12', '13', '14', '15',
         '2', '3', '4', '5', '6', '7', '8', '9']


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "datasets/videos/video2.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 69

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Lettura frame {frame_idx} fallita")

    overlay = frame.copy()

    # --- TAVOLO ---
    mask, med = cloth_mask(frame)
    print(f"Panno HSV mediano: {med.astype(int)}")
    corners = order_corners(cv2.boxPoints(cv2.minAreaRect(largest_contour(mask))))
    cv2.polylines(overlay, [corners.astype(int)], True, (0, 255, 255), 2)

    # --- BUCHE ---
    pockets = find_pockets(mask, corners)
    print(f"Buche trovate: {len(pockets)}")
    for (p, r, area) in pockets:
        cv2.circle(overlay, (int(p[0]), int(p[1])), 30, (0, 0, 255), 3)
        cv2.circle(overlay, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

    # --- PALLE (YOLO native, con dedup dei box quasi-duplicati da TTA) ---
    model = YOLO(WEIGHTS)
    r = model.predict(frame, conf=0.25, augment=True, verbose=False)[0]
    dets = sorted([(b.xyxy[0].int().tolist(), int(b.cls[0]), float(b.conf[0]))
                   for b in r.boxes], key=lambda t: t[2], reverse=True)
    kept = []
    for (x1, y1, x2, y2), c, cf in dets:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(kx1 <= cx <= kx2 and ky1 <= cy <= ky2
               for (kx1, ky1, kx2, ky2), _, _ in kept):
            continue                      # centro dentro un box gia' tenuto -> duplicato
        kept.append(((x1, y1, x2, y2), c, cf))
    labels = [NAMES[c] for (_, c, _) in kept]
    uconf = [cf for (_, _, cf) in kept]   # confidenza usata per la priorita' di unicita'

    # Regola UNA SOLA battente: se piu' palle danno '0', tengo come battente la
    # piu' bianca; le altre le rietichetto col COLORE (forzando un non-battente).
    cue_idx = [i for i, l in enumerate(labels) if l == "0"]
    if len(cue_idx) > 1:
        feats = {i: ball_features(frame[kept[i][0][1]:kept[i][0][3],
                                        kept[i][0][0]:kept[i][0][2]]) for i in cue_idx}
        keep = max(cue_idx, key=lambda i: feats[i]["wf"])
        for i in cue_idx:
            if i != keep:
                labels[i] = label_from_features(feats[i], allow_cue=False)
                uconf[i] = 0.0            # identita' da colore su una declassata = inaffidabile
        print(f"Due battenti: tengo box#{keep} (piu' bianca), l'altra -> colore")

    # VINCOLO UNICITA': ogni numero (tranne 0) al piu' una volta. Processo per
    # confidenza decrescente: chi arriva dopo su un numero gia' preso -> '?'.
    seen = set()
    for i in sorted(range(len(kept)), key=lambda i: uconf[i], reverse=True):
        lab = labels[i]
        if lab in ("0", "?"):
            continue
        if lab in seen:
            labels[i] = "?"
        else:
            seen.add(lab)

    # RECUPERO '?': la NMS di YOLO e' per-classe, quindi a conf bassa la stessa
    # palla riappare sotto classi diverse. Per ogni '?' prendo, in quel punto,
    # la classe a confidenza piu' alta ancora LIBERA (rispetta l'unicita').
    q_idx = [i for i, l in enumerate(labels) if l == "?"]
    if q_idx:
        r2 = model.predict(frame, conf=0.05, augment=False, verbose=False)[0]
        alt = sorted([(b.xyxy[0].int().tolist(), int(b.cls[0]), float(b.conf[0]))
                      for b in r2.boxes], key=lambda t: t[2], reverse=True)
        for i in q_idx:
            (x1, y1, x2, y2), _, _ = kept[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for (ax1, ay1, ax2, ay2), ac, acf in alt:
                name = NAMES[ac]
                if ax1 <= cx <= ax2 and ay1 <= cy <= ay2 and name != "0" and name not in seen:
                    labels[i] = name
                    seen.add(name)
                    print(f"  '?' recuperata -> {name} (2a ipotesi YOLO, conf {acf:.2f})")
                    break

    for ((x1, y1, x2, y2), c, cf), lab in zip(kept, labels):
        col = (0, 255, 0) if lab == "0" else (0, 200, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
        cv2.putText(overlay, f"{lab} {cf:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    print(f"Palle rilevate: {len(kept)} (dopo dedup, da {len(dets)})")

    out = "output/core_overlay.png"
    cv2.imwrite(out, overlay)
    # salvo anche la maschera per diagnosi
    cv2.imwrite("output/core_mask.png", mask)
    print(f"Salvato {out}")


if __name__ == "__main__":
    main()
