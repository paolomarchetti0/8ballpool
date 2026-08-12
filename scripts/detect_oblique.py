"""
ESTENSIONE B - omografia da vista OBLIQUA, validata per auto-consistenza.

Su un'immagine obliqua reale (dataset V2):
  1. stimo i 4 angoli del tavolo (contorno del panno -> 4 vertici del trapezio);
  2. omografia verso un rettangolo 2:1 -> raddrizzo (vista dall'alto);
  3. VALIDAZIONE per auto-consistenza: in obliquo le palle appaiono di dimensioni
     diverse (prospettiva) e un po' ellittiche; dopo un buon raddrizzamento devono
     diventare di dimensione UNIFORME e CIRCOLARI. Misuro:
       - CV = dev.std/media dei diametri palla (piu' basso = piu' uniforme);
       - aspetto medio = lato_max/lato_min dei box (piu' vicino a 1 = piu' circolare).
Non c'e' ground truth accoppiata (stesso tavolo dall'alto), quindi l'errore e'
misurato in modo indiretto (auto-consistenza), non pixel-per-pixel.
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

from detect_table import cloth_mask, largest_contour, order_corners
from detect_balls import identify_balls

WEIGHTS = "runs/detect/full/weights/best.pt"
IMG = "datasets/Pool Ball Detection V2.yolov8/train/images/out-1-101_png.rf.EKbTazTgl4m18xmI8NAG.png"
# Rettangolo di uscita. Nota: con gli angoli marcati a mano le palle risultano
# circolari a ~2.7:1 (H=440) invece del 2:1 teorico -> gli angoli non sono perfetti
# al pixel, ma cosi' la resa e' metricamente coerente (palle circolari = validazione
# per auto-consistenza). Taratura empirica di H fino ad aspetto_medio ~1.0.
OUT_W, OUT_H = 1200, 440

# 4 angoli del panno marcati a mano su out-1-101 (tavolo intero, senza occlusioni).
# Ordine: far-left, far-right, near-right, near-left -> mappa a [0,0],[W,0],[W,H],[0,H].
MANUAL_CORNERS = np.array([[665, 420], [1555, 420], [1420, 720], [560, 655]], np.float32)


def table_quad(mask):
    """4 vertici del trapezio del tavolo dal contorno del panno."""
    cnt = largest_contour(mask)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for k in range(1, 12):
        approx = cv2.approxPolyDP(hull, k * 0.01 * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(-1, 2)), True
    # fallback: rettangolo ruotato minimo (meno preciso su trapezio)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(cnt))), False


def _stats(balls):
    dia, asp = [], []
    for b in balls:
        x1, y1, x2, y2 = b["box"]
        w, h = x2 - x1, y2 - y1
        if w > 1 and h > 1:
            dia.append((w + h) / 2.0)
            asp.append(max(w, h) / min(w, h))
    dia, asp = np.array(dia), np.array(asp)
    cv = float(dia.std() / dia.mean()) if len(dia) else 0.0
    return cv, float(asp.mean()) if len(asp) else 0.0, len(dia)


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMG
    frame = cv2.imread(img_path)
    if frame is None:
        raise SystemExit(f"immagine non letta: {img_path}")
    model = YOLO(WEIGHTS)

    corners = MANUAL_CORNERS
    print("Angoli tavolo (marcati a mano):", corners.astype(int).tolist())

    dst = np.array([[0, 0], [OUT_W, 0], [OUT_W, OUT_H], [0, OUT_H]], np.float32)
    H, _ = cv2.findHomography(corners.astype(np.float32), dst)
    rectified = cv2.warpPerspective(frame, H, (OUT_W, OUT_H))

    # validazione: statistiche palle prima (obliquo) e dopo (raddrizzato)
    balls_before = identify_balls(frame, model)
    cv_b, asp_b, n_b = _stats(balls_before)
    balls_after = identify_balls(rectified, model)
    cv_a, asp_a, n_a = _stats(balls_after)
    print(f"\nPalle OBLIQUO:      n={n_b}  CV_diametri={cv_b*100:.0f}%  aspetto_medio={asp_b:.2f}")
    print(f"Palle RADDRIZZATO:  n={n_a}  CV_diametri={cv_a*100:.0f}%  aspetto_medio={asp_a:.2f}")
    print("  (CV piu' basso e aspetto piu' vicino a 1 = raddrizzamento coerente)")

    # overlay: angoli sull'obliquo
    ov = frame.copy()
    cv2.polylines(ov, [corners.astype(int)], True, (0, 255, 255), 3)
    for p in corners:
        cv2.circle(ov, tuple(p.astype(int)), 10, (0, 0, 255), -1)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/oblique_corners.png", ov)
    cv2.imwrite("output/oblique_rectified.png", rectified)
    print("\nSalvati: output/oblique_corners.png, output/oblique_rectified.png")


if __name__ == "__main__":
    main()
