"""
ESTENSIONE C - validazione dell'esito.

Confronta la predizione fatta sul frame del tiro con quello che succede DAVVERO
nei frame successivi (usati SOLO per validare, non per predire):
  1. predico sul frame del tiro -> esito IN/OUT + quale palla;
  2. su un frame DOPO (a palle ferme) rilevo le palle rimaste;
  3. le palle SPARITE dal tavolo = palle imbucate davvero;
  4. confronto: se avevo detto "IN: palla X", la X dovrebbe essere sparita;
     se "OUT", non dovrebbe sparire nulla.

Uso:
  python scripts/validate.py "datasets/videos/video2.mp4" 69           # after-frame automatico
  python scripts/validate.py "datasets/videos/video2.mp4" 69 300       # after-frame esplicito

Nota onesta: identita'/detection hanno rumore, e una palla imbucata puo' restare
visibile nella retina della buca -> conto solo le palle in GIOCO APERTO (lontane
dalle buche). E' una validazione approssimata, non una ground-truth perfetta.
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

from detect_table import cloth_mask, largest_contour, order_corners, TOP_W, TOP_H
from detect_balls import identify_balls
from predict import run_prediction, POCKETS_TOP, draw_ball_label

WEIGHTS = "runs/detect/full/weights/best.pt"


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"frame {idx} non letto")
    return fr


def last_still_frame(video, after=0.5):
    """Ultimo frame 'fermo' (motion~0) nella parte finale del video (a palle ferme)."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev, best = None, None
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2GRAY)
        if prev is not None and i > after * n:
            if float(np.mean(cv2.absdiff(g, prev))) < 0.5:
                best = i           # tengo l'ultimo frame fermo
        prev = g
        i += 1
    cap.release()
    return best if best is not None else n - 1


def open_play_labels(frame, model):
    """Etichette delle palle in GIOCO APERTO (lontane dalle buche), + quante."""
    mask, _ = cloth_mask(frame)
    corners = order_corners(cv2.boxPoints(cv2.minAreaRect(largest_contour(mask))))
    dst = np.array([[0, 0], [TOP_W, 0], [TOP_W, TOP_H], [0, TOP_H]], np.float32)
    H, _ = cv2.findHomography(corners, dst)
    balls = identify_balls(frame, model)
    labels = []
    for b in balls:
        if b["label"] in ("0", "?"):
            continue
        top = cv2.perspectiveTransform(np.array([[b["center"]]], np.float32), H)[0, 0]
        if min(np.linalg.norm(top - np.array(q)) for q in POCKETS_TOP) < 40:
            continue               # dentro/vicino a una buca -> non in gioco aperto
        labels.append(b["label"])
    return balls, set(labels)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "datasets/videos/video2.mp4"
    shot_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 69
    model = YOLO(WEIGHTS)

    # 1. predizione sul frame del tiro
    frame = read_frame(video, shot_frame)
    pred = run_prediction(frame, model)
    if pred is None:
        raise SystemExit("predizione fallita (battente/stecca)")
    print(f"PREDIZIONE (frame {shot_frame}): catena battente -> {' -> '.join(pred['chain'])}")
    print(f"  esito predetto: {pred['result']}"
          + (f" (palla {pred['potted_label']})" if pred['potted_label'] else ""))

    before = {b["label"] for b in pred["balls"] if b["label"] not in ("0", "?")}

    # 2. frame DOPO (a palle ferme)
    after_frame = int(sys.argv[3]) if len(sys.argv) > 3 else last_still_frame(video)
    fr_after = read_frame(video, after_frame)
    balls_after, after = open_play_labels(fr_after, model)
    print(f"\nOSSERVATO (frame {after_frame}):")
    print(f"  palle in gioco PRIMA: {len(before)}  DOPO: {len(after)}")
    disappeared = before - after
    print(f"  palle SPARITE dal tavolo: {sorted(disappeared) if disappeared else 'nessuna'}")

    # 3. verdetto
    if pred["result"] == "IN":
        ok = pred["potted_label"] in disappeared
        extra = "" if ok else "  (la palla predetta non risulta sparita)"
        verdict = "COERENTE" if ok else "NON coerente"
        print(f"\nVERDETTO: predetto IN palla {pred['potted_label']} -> {verdict}{extra}")
    elif pred["result"] == "OUT":
        ok = len(disappeared) == 0
        verdict = "COERENTE" if ok else f"NON coerente (sparite: {sorted(disappeared)})"
        print(f"\nVERDETTO: predetto OUT -> {verdict}")
    else:
        print(f"\nVERDETTO: predetto {pred['result']} (validazione non gestita)")

    # overlay del frame DOPO con le palle rilevate
    for b in balls_after:
        col = (255, 255, 255) if b["label"] == "0" else (0, 255, 255)
        draw_ball_label(fr_after, b["center"], b["label"], col)
    name = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
    out = f"output/validate_{name}_f{shot_frame}_vs_f{after_frame}.png"
    cv2.imwrite(out, fr_after)
    print(f"\nSalvato {out}")


if __name__ == "__main__":
    main()
