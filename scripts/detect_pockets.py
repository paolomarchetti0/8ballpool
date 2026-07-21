"""
Buche - 6 posizioni sul tavolo.

Metodo: DATA-DRIVEN dalla maschera del panno (niente offset "a occhio").
  Nella maschera del panno le buche sono i MORSI CONCAVI del bordo (4 angoli
  arrotondati + 2 semicerchi a meta' dei lati lunghi). Le palle invece sono
  buchi INTERNI e non toccano il bordo.

  Idea: convex hull del panno MENO il panno = esattamente le zone concave,
  cioe' le 6 buche. Di ognuna prendo il baricentro (centro buca) e il raggio
  equivalente (raggio di cattura). Cosi' il centro cade dentro la ZONA SCURA
  della buca, come deve essere.

Output in output/:
  - pockets_overlay.png  : frame con tavolo + 6 buche
  - pockets_topdown.png  : vista dall'alto con le 6 buche
"""
import os
import cv2
import numpy as np

from detect_table import (
    VIDEO, FRAME_IDX, OUT_DIR, TOP_W, TOP_H,
    read_frame, cloth_mask, largest_contour, order_corners,
)

MIN_POCKET_AREA = 400    # area minima (px^2) per considerare una rientranza una buca
MAX_POCKET_AREA = 6000   # area massima: sopra e' occlusione (corpo/braccio giocatore), non una buca
MATCH_DIST = 140         # px: distanza max per agganciare un blob alla posizione attesa
CLOSE_K = 121            # kernel della chiusura morfologica (~ diametro buca in px)
# Raggio di cattura: la palla e' imbucata se il suo CENTRO passa entro questo
# raggio dal centro buca. Fisicamente ~ (semi-larghezza bocca - raggio palla),
# quindi PIU' PICCOLO della bocca (un cerchio troppo grande = falsi positivi:
# palle che sfiorano la buca contate come imbucate). Provvisorio: da calibrare
# col raggio palla vero (dallo YOLO).
CAPTURE_R = 30           # px


def geometric_positions(corners):
    """6 posizioni attese: 4 angoli + 2 meta' dei lati lunghi (alto/basso)."""
    tl, tr, br, bl = corners
    return np.array([tl, tr, br, bl, (tl + tr) / 2, (bl + br) / 2], dtype=np.float32)


def find_pockets(mask, corners):
    """Trova le 6 buche in modo ROBUSTO all'occlusione (giocatore).
    Le buche sono "baie" concave del bordo del panno (chiusura - panno). Ma un
    giocatore appoggiato crea una baia enorme: la scarto per AREA. I blob rimasti
    li AGGANCIO alle 6 posizioni attese (angoli + meta' lati); dove una buca e'
    occlusa e non c'e' blob vicino, uso la posizione geometrica.
    Restituisce lista di (centro_xy, raggio_equivalente, area)."""
    cnt = largest_contour(mask)

    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [cnt], -1, 255, cv2.FILLED)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    closed = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k)
    diff = cv2.subtract(closed, filled)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # blob candidati di dimensione da buca (scarto rumore e occlusioni grandi)
    cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] == 0 or not (MIN_POCKET_AREA <= area <= MAX_POCKET_AREA):
            continue
        blobs.append((np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], np.float32),
                      float(np.sqrt(area / np.pi)), area))

    # aggancio ogni posizione attesa al blob piu' vicino (entro MATCH_DIST)
    geo = geometric_positions(corners)
    pockets = []
    for g in geo:
        best, bestd = None, MATCH_DIST
        for (c, r, a) in blobs:
            d = float(np.linalg.norm(c - g))
            if d < bestd:
                best, bestd = (c, r, a), d
        if best is not None:
            pockets.append(best)                     # buca vista: uso la zona scura
        else:
            pockets.append((g.astype(np.float32), 25.0, 0.0))  # occlusa: geometrica
    return pockets


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frame = read_frame(VIDEO, FRAME_IDX)
    mask, _ = cloth_mask(frame)
    corners = order_corners(cv2.boxPoints(cv2.minAreaRect(largest_contour(mask))))

    pockets = find_pockets(mask, corners)
    print(f"Buche trovate: {len(pockets)}")
    for i, (p, r, area) in enumerate(pockets):
        print(f"  #{i}: centro=({p[0]:.0f}, {p[1]:.0f})  raggio~{r:.0f}px  area={area:.0f}")

    centers = np.array([p for p, _, _ in pockets], dtype=np.float32)
    radii = [r for _, r, _ in pockets]

    # Omografia -> vista dall'alto, e buche riproiettate.
    dst = np.array([[0, 0], [TOP_W, 0], [TOP_W, TOP_H], [0, TOP_H]], dtype=np.float32)
    H, _ = cv2.findHomography(corners, dst)
    topdown = cv2.warpPerspective(frame, H, (TOP_W, TOP_H))
    centers_td = cv2.perspectiveTransform(centers.reshape(-1, 1, 2), H).reshape(-1, 2)

    # --- Disegni: cerchio = raggio di cattura, punto = centro buca ---
    overlay = frame.copy()
    cv2.polylines(overlay, [corners.astype(int)], True, (0, 255, 255), 2)
    for p in centers:
        cv2.circle(overlay, (int(p[0]), int(p[1])), CAPTURE_R, (0, 0, 255), 3)
        cv2.circle(overlay, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
    for p in centers_td:
        cv2.circle(topdown, (int(p[0]), int(p[1])), CAPTURE_R, (0, 0, 255), 3)
        cv2.circle(topdown, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(OUT_DIR, "pockets_overlay.png"), overlay)
    cv2.imwrite(os.path.join(OUT_DIR, "pockets_topdown.png"), topdown)
    print("\nSalvati: output/pockets_overlay.png, output/pockets_topdown.png")


if __name__ == "__main__":
    main()
