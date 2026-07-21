"""
Punto 4 - Detection del TAVOLO (solo tavolo, metodo classico).

Idea (tutta CV classica, niente ML):
  1. Segmento il panno per COLORE (il ciano e' molto saturo e uniforme) -> maschera.
  2. Chiudo i buchi (palle, buche) con morfologia e tengo il blob piu' grande.
  3. Dal blob ricavo i 4 ANGOLI del tavolo (rettangolo ruotato minimo).
  4. Con i 4 angoli stimo l'OMOGRAFIA verso un rettangolo canonico
     -> "vista dall'alto" rettificata, dove faremo la geometria dei tiri.

Output in output/:
  - table_overlay.png : frame originale con maschera, contorno e 4 angoli disegnati
  - table_topdown.png : vista dall'alto rettificata (warp)
"""
import os
import cv2
import numpy as np

VIDEO = "datasets/videos/pool 1.mp4"
FRAME_IDX = 72          # frame pulito (palle ferme, niente stecca in mezzo)
OUT_DIR = "output"

# Dimensioni della vista dall'alto rettificata.
# Un tavolo da biliardo ha proporzioni ~2:1 (lunghezza : larghezza).
TOP_W, TOP_H = 1000, 500


def read_frame(video, idx):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Non riesco ad aprire il video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Lettura frame {idx} fallita")
    return frame


def cloth_mask(frame_bgr):
    """Maschera del panno: soglia HSV attorno al colore prevalente al centro."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]

    # Campiono un rettangolo centrale (sicuramente panno) per stimare il colore.
    cx0, cx1 = int(w * 0.40), int(w * 0.60)
    cy0, cy1 = int(h * 0.40), int(h * 0.60)
    patch = hsv[cy0:cy1, cx0:cx1].reshape(-1, 3)
    med = np.median(patch, axis=0)

    # Range attorno alla mediana: largo su S/V (ombre/riflessi), stretto su H.
    tol = np.array([15, 80, 90])
    lower = np.clip(med - tol, [0, 30, 30], [179, 255, 255]).astype(np.uint8)
    upper = np.clip(med + tol, [0, 30, 30], [179, 255, 255]).astype(np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morfologia: chiudo i buchi (palle, buche, stecca) e pulisco il rumore.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask, med


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise SystemExit("Nessun contorno trovato nella maschera del panno")
    return max(cnts, key=cv2.contourArea)


def order_corners(pts):
    """Ordina 4 punti in: alto-sx, alto-dx, basso-dx, basso-sx."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # alto-sx  (x+y minimo)
        pts[np.argmin(d)],   # alto-dx  (x-y massimo -> y-x minimo)
        pts[np.argmax(s)],   # basso-dx (x+y massimo)
        pts[np.argmax(d)],   # basso-sx (x-y minimo -> y-x massimo)
    ], dtype=np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frame = read_frame(VIDEO, FRAME_IDX)
    mask, med = cloth_mask(frame)
    print(f"Colore panno stimato (HSV mediano): {med.astype(int)}")

    cnt = largest_contour(mask)
    area_ratio = cv2.contourArea(cnt) / (frame.shape[0] * frame.shape[1])
    print(f"Area del panno: {area_ratio*100:.1f}% del frame")

    # 4 angoli = rettangolo ruotato minimo che racchiude il panno.
    rect = cv2.minAreaRect(cnt)
    corners = order_corners(cv2.boxPoints(rect))
    print("Angoli tavolo (px):")
    for name, (x, y) in zip(["alto-sx", "alto-dx", "basso-dx", "basso-sx"], corners):
        print(f"  {name:9s}: ({x:.0f}, {y:.0f})")

    # Omografia verso il rettangolo canonico (vista dall'alto).
    dst = np.array([[0, 0], [TOP_W, 0], [TOP_W, TOP_H], [0, TOP_H]], dtype=np.float32)
    H, _ = cv2.findHomography(corners, dst)
    topdown = cv2.warpPerspective(frame, H, (TOP_W, TOP_H))

    # --- Overlay diagnostico sul frame originale ---
    overlay = frame.copy()
    # maschera del panno in verde semitrasparente
    green = np.zeros_like(frame); green[mask > 0] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, green, 0.25, 0)
    # contorno del tavolo
    cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)
    # bordo tavolo (i 4 angoli) e pallini
    cv2.polylines(overlay, [corners.astype(int)], True, (0, 255, 255), 3)
    for (x, y) in corners:
        cv2.circle(overlay, (int(x), int(y)), 12, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(OUT_DIR, "table_overlay.png"), overlay)
    cv2.imwrite(os.path.join(OUT_DIR, "table_topdown.png"), topdown)
    cv2.imwrite(os.path.join(OUT_DIR, "table_mask.png"), mask)
    print("\nSalvati: output/table_overlay.png, output/table_topdown.png, output/table_mask.png")


if __name__ == "__main__":
    main()
