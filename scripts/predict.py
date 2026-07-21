"""
PREDIZIONE geometrica - versione 2: primo impatto (ghost-ball).

1. La BATTENTE viaggia lungo la direzione della stecca, rimbalzando sulle sponde,
   finche' incontra la PRIMA palla (o cade in buca, o si esauriscono i rimbalzi).
2. Al contatto (ghost-ball): la palla colpita parte lungo la CONGIUNGENTE DEI
   CENTRI (dal centro della battente al contatto verso il centro della palla).
3. Simulo la traiettoria della PALLA COLPITA (con rimbalzi) verso l'eventuale buca
   -> esito IN / OUT.

Tutto in vista dall'alto rettificata (rimbalzi = riflessioni), poi riproietto.
Collisione tra due palle di raggio R: i centri si toccano a distanza 2R.
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

from detect_table import cloth_mask, largest_contour, order_corners, TOP_W, TOP_H
from detect_cue import detect_cue
from detect_balls import identify_balls

WEIGHTS = "runs/detect/full/weights/best.pt"
MAX_BOUNCES = 12
POCKETS_TOP = [(0, 0), (TOP_W, 0), (TOP_W, TOP_H), (0, TOP_H),
               (TOP_W / 2, 0), (TOP_W / 2, TOP_H)]


def draw_ball_label(ov, center, label, color):
    """Disegna il numero della palla sopra di essa."""
    x, y = int(center[0]), int(center[1])
    cv2.putText(ov, label, (x - 10, y - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 4)
    cv2.putText(ov, label, (x - 10, y - 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def nearest_wall(P, d, R):
    xmin, xmax, ymin, ymax = R, TOP_W - R, R, TOP_H - R
    ts = []
    if d[0] > 1e-9:  ts.append(((xmax - P[0]) / d[0], "x"))
    if d[0] < -1e-9: ts.append(((xmin - P[0]) / d[0], "x"))
    if d[1] > 1e-9:  ts.append(((ymax - P[1]) / d[1], "y"))
    if d[1] < -1e-9: ts.append(((ymin - P[1]) / d[1], "y"))
    ts = [(t, ax) for t, ax in ts if t > 1e-6]
    return min(ts) if ts else (None, None)


def first_ball_hit(P, d, R, balls, t_max):
    """Prima palla colpita lungo P+t d entro t_max. Contatto a distanza 2R tra centri."""
    best_t, best_i = None, None
    for i, C in enumerate(balls):
        f = P - C
        b = float(f @ d)
        c = float(f @ f) - (2 * R) ** 2
        disc = b * b - c
        if disc < 0:
            continue
        t = -b - np.sqrt(disc)          # prima radice (avvicinamento)
        if 1e-6 < t < t_max and (best_t is None or t < best_t):
            best_t, best_i = t, i
    return best_t, best_i


def simulate_cue(P, d, R, balls, capture):
    """Traiettoria battente fino alla prima palla / buca. Ritorna (path, event)."""
    pockets = [np.array(q, np.float64) for q in POCKETS_TOP]
    P = P.astype(np.float64); d = d / (np.linalg.norm(d) + 1e-9)
    path = [P.copy()]
    for _ in range(MAX_BOUNCES):
        t_wall, ax = nearest_wall(P, d, R)
        if t_wall is None:
            break
        t_ball, i = first_ball_hit(P, d, R, balls, t_wall)
        if t_ball is not None:                        # colpisce una palla
            contact = P + d * t_ball
            path.append(contact)
            struck_dir = balls[i] - contact
            return path, ("ball", i, contact, struck_dir / (np.linalg.norm(struck_dir) + 1e-9))
        hit = P + d * t_wall
        path.append(hit)
        if min(np.linalg.norm(hit - q) for q in pockets) <= capture:
            return path, ("pocket_cue",)             # battente in buca (fallo)
        d = np.array([-d[0], d[1]]) if ax == "x" else np.array([d[0], -d[1]])
        P = hit
    return path, ("none",)


def simulate_ball(P, d, R, capture):
    """PRIMA RETTA della palla colpita: dal suo centro fino alla prima sponda.
    IN se quel punto e' su una buca (tiro diretto), altrimenti OUT. (path, potted)."""
    pockets = [np.array(q, np.float64) for q in POCKETS_TOP]
    P = P.astype(np.float64); d = d / (np.linalg.norm(d) + 1e-9)
    t_wall, _ = nearest_wall(P, d, R)
    if t_wall is None:
        return [P], False
    hit = P + d * t_wall
    potted = min(np.linalg.norm(hit - q) for q in pockets) <= capture
    return [P, hit], potted


def draw_path(ov, path_top, Hinv, color, thick=3):
    pi = cv2.perspectiveTransform(np.array([path_top], np.float32), Hinv)[0]
    for a, b in zip(pi[:-1], pi[1:]):
        cv2.line(ov, tuple(a.astype(int)), tuple(b.astype(int)), color, thick)


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
    mask, _ = cloth_mask(frame)
    corners = order_corners(cv2.boxPoints(cv2.minAreaRect(largest_contour(mask))))
    dst = np.array([[0, 0], [TOP_W, 0], [TOP_W, TOP_H], [0, TOP_H]], np.float32)
    H, _ = cv2.findHomography(corners, dst)
    Hinv = np.linalg.inv(H)

    def to_top(pt):
        return cv2.perspectiveTransform(np.array([[pt]], np.float32), H)[0, 0].astype(np.float64)

    balls = identify_balls(frame, model)
    cue = max((b for b in balls if b["label"] == "0"), key=lambda b: b["conf"], default=None)
    if cue is None:
        raise SystemExit("battente non trovata")
    cue_img, R_img = cue["center"], cue["radius"]
    _, shot_img = detect_cue(frame, model, ball=cue_img)
    if shot_img is None:
        raise SystemExit("stecca non trovata")

    # object balls (tutte tranne la battente) in vista dall'alto
    obj = [b for b in balls if b is not cue]
    obj_top = [to_top(b["center"]) for b in obj]
    P0 = to_top(cue_img)
    d0 = to_top(cue_img + shot_img * 200) - P0
    R_top = float(np.linalg.norm(to_top(cue_img + np.array([R_img, 0], np.float32)) - P0))
    capture = 2.2 * R_top

    cue_path, event = simulate_cue(P0, d0, R_top, obj_top, capture)

    ov = frame.copy()
    for q in POCKETS_TOP:
        c = cv2.perspectiveTransform(np.array([[q]], np.float32), Hinv)[0, 0]
        cv2.circle(ov, tuple(c.astype(int)), int(capture), (0, 255, 255), 1)

    # numero su OGNI palla
    for b in balls:
        col = (255, 255, 255) if b["label"] == "0" else (0, 255, 255)
        draw_ball_label(ov, b["center"], b["label"], col)

    draw_path(ov, cue_path, Hinv, (0, 0, 255), 3)                 # battente = rosso
    cv2.circle(ov, tuple(cue_img.astype(int)), int(R_img), (255, 255, 255), 2)

    result, potted_label = "OUT", None
    if event[0] == "ball":
        i, contact, sdir = event[1], event[2], event[3]
        struck_path, potted = simulate_ball(obj_top[i], sdir, R_top, capture)
        draw_path(ov, struck_path, Hinv, (0, 200, 0), 3)         # palla colpita = verde
        cv2.circle(ov, tuple(obj[i]["center"].astype(int)), int(R_img) + 4, (0, 200, 0), 3)
        if potted:
            result, potted_label = "IN", obj[i]["label"]
        print(f"Prima palla colpita: {obj[i]['label']}  ->  "
              f"{'IMBUCATA' if potted else 'NON imbucata'}")
    elif event[0] == "pocket_cue":
        result = "SCRATCH"
        print("La battente finisce in buca (fallo)")
    else:
        print("Nessuna palla colpita")

    txt = result if potted_label is None else f"{result}: palla {potted_label}"
    cv2.putText(ov, txt, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0, 255, 0) if result == "IN" else (0, 0, 255), 5)
    name = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
    out = f"output/predict_{name}_f{frame_idx}.png"
    cv2.imwrite(out, ov)
    print(f"Esito: {txt}  |  salvato {out}")


if __name__ == "__main__":
    main()
