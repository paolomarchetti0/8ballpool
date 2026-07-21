"""
Detection STECCA (classico, Hough) sul frame "prima del colpo".

Idea:
  1. Trovo la BATTENTE (bianca) con lo YOLO -> la userò come vincolo.
  2. ROI = interno del tavolo (maschera panno erosa: evito le sponde).
  3. Canny + HoughLinesP -> tanti segmenti.
  4. La stecca e' una LINEA LUNGA che PASSA VICINO alla battente (le sponde no,
     le altre linee no). Tengo i segmenti lunghi con distanza dalla bianca piccola
     e fitto un'unica retta (l'asse della stecca).
  5. VERSO: la stecca sta DIETRO la bianca; il colpo va dalla stecca -> bianca e
     PROSEGUE oltre. Direzione = (centro_bianca - baricentro_punti_stecca).

Output: output/cue_overlay.png con asse stecca + freccia della direzione del tiro.
"""
import sys
import cv2
import numpy as np
from ultralytics import YOLO

from detect_table import cloth_mask

WEIGHTS = "runs/detect/full/weights/best.pt"

MIN_LEN = 60        # lunghezza minima segmento (px)
MAX_DIST_BALL = 35  # distanza max (px) tra la retta del segmento e il centro bianca
ROI_ERODE = 10      # erosione della maschera tavolo per stare lontano dalle sponde


def cue_ball_center(frame, model):
    """Centro della battente (classe '0' YOLO piu' confidente)."""
    r = model.predict(frame, conf=0.25, augment=True, verbose=False)[0]
    best, bestcf = None, -1
    for b in r.boxes:
        if int(b.cls[0]) == 0 and float(b.conf[0]) > bestcf:   # indice 0 = '0' = bianca
            x1, y1, x2, y2 = b.xyxy[0].int().tolist()
            best, bestcf = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32), float(b.conf[0])
    return best


def point_line_dist(pt, a, b):
    """Distanza punto pt dalla retta passante per a,b."""
    a, b, pt = map(lambda p: np.asarray(p, np.float32), (a, b, pt))
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-6:
        return np.linalg.norm(pt - a)
    return abs(np.cross(d, pt - a)) / n


def detect_cue(frame, model, ball=None):
    """Rileva la stecca e restituisce (ball, shot_dir) in coordinate immagine.
    ball = centro battente (se None lo cerca con lo YOLO)."""
    if ball is None:
        ball = cue_ball_center(frame, model)
    if ball is None:
        return None, None

    # ROI interno tavolo
    mask, _ = cloth_mask(frame)
    roi = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (ROI_ERODE, ROI_ERODE)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=roi)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=MIN_LEN, maxLineGap=25)
    pts = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = np.hypot(x2 - x1, y2 - y1)
            if length < MIN_LEN:
                continue
            if point_line_dist(ball, (x1, y1), (x2, y2)) <= MAX_DIST_BALL:
                # campiono punti LUNGO il segmento: i segmenti lunghi pesano di
                # piu' -> angolo del fit piu' preciso (leva lunga = meno errore).
                n = max(2, int(length / 8))
                for k in range(n):
                    s = k / (n - 1)
                    pts.append((x1 + (x2 - x1) * s, y1 + (y2 - y1) * s))
    if not pts:
        return ball, None
    pts = np.array(pts, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    axis = np.array([vx, vy], np.float32)
    if np.dot(ball - pts.mean(axis=0), axis) < 0:   # verso: via dalla stecca
        axis = -axis
    return ball, axis / (np.linalg.norm(axis) + 1e-6)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "datasets/videos/video3.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("frame non letto")

    model = YOLO(WEIGHTS)
    ball = cue_ball_center(frame, model)
    if ball is None:
        raise SystemExit("battente non trovata")
    print(f"Battente: ({ball[0]:.0f}, {ball[1]:.0f})")

    # ROI interno tavolo
    mask, _ = cloth_mask(frame)
    roi = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                    (ROI_ERODE, ROI_ERODE)))

    # Canny + Hough dentro la ROI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=roi)
    cv2.imwrite("output/cue_edges.png", edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=MIN_LEN, maxLineGap=25)
    print(f"Linee Hough totali: {0 if lines is None else len(lines)}")

    # segmenti lunghi che passano vicino alla battente
    pts = []
    cand = 0
    dbg = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = np.hypot(x2 - x1, y2 - y1)
            if length < MIN_LEN:
                continue
            d = point_line_dist(ball, (x1, y1), (x2, y2))
            dbg.append((length, d))
            if d <= MAX_DIST_BALL:
                pts.extend([(x1, y1), (x2, y2)])
                cand += 1
    dbg.sort(reverse=True)
    print("Top segmenti (len, dist_bianca):", [(int(l), int(d)) for l, d in dbg[:8]])
    print(f"Segmenti candidati stecca: {cand}")
    if not pts:
        raise SystemExit("stecca non trovata")

    pts = np.array(pts, np.float32)
    # asse della stecca = retta ai minimi quadrati sui punti dei segmenti
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    axis = np.array([vx, vy], np.float32)

    # VERSO: la stecca sta dietro la bianca; direzione tiro = bianca - baricentro punti
    centroid = pts.mean(axis=0)
    shot = ball - centroid
    if np.dot(shot, axis) < 0:
        axis = -axis
    shot_dir = axis / (np.linalg.norm(axis) + 1e-6)
    ang = np.degrees(np.arctan2(shot_dir[1], shot_dir[0]))
    print(f"Direzione tiro: ({shot_dir[0]:.2f}, {shot_dir[1]:.2f})  angolo {ang:.1f} deg")

    # --- disegni ---
    ov = frame.copy()
    # asse stecca (linea gialla lunga)
    p1 = (int(x0 - vx * 1500), int(y0 - vy * 1500))
    p2 = (int(x0 + vx * 1500), int(y0 + vy * 1500))
    cv2.line(ov, p1, p2, (0, 255, 255), 1)
    # battente
    cv2.circle(ov, tuple(ball.astype(int)), 6, (255, 255, 255), -1)
    # freccia direzione tiro dalla bianca
    tip = (ball + shot_dir * 350).astype(int)
    cv2.arrowedLine(ov, tuple(ball.astype(int)), tuple(tip), (0, 0, 255), 4, tipLength=0.06)
    cv2.imwrite("output/cue_overlay.png", ov)
    cv2.imwrite("output/cue_edges.png", edges)
    print("Salvato output/cue_overlay.png")


if __name__ == "__main__":
    main()
