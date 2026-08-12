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

from detect_table import cloth_mask, largest_contour, order_corners, cushion_corners, TOP_W, TOP_H
from detect_cue import detect_cue
from detect_balls import identify_balls

WEIGHTS = "runs/detect/full/weights/best.pt"
MAX_BOUNCES = 12
# NOTA (provato e scartato): il cappello sponda (stesso ciano del campo) fa
# sbordare minAreaRect di ~pochi px oltre il naso, e nella mappa 2D una palla a
# ridosso della sponda appare staccata. Abbiamo provato a rimpicciolire il
# rettangolo (inset fisso, anche per-lato) e a stimare il naso dall'ombra: MA
# per la SENSIBILITA' del predittore anche un inset del 2% sposta le buche quel
# tanto che basta a ribaltare esiti VALIDATI (video3 OUT -> IN falso). minAreaRect
# e' l'unico che azzecca video2 (IN 4) e video3 (OUT). Quindi si tiene minAreaRect
# e si accetta la piccola imprecisione della mappa (limite onesto documentato).
POCKETS_TOP = [(0, 0), (TOP_W, 0), (TOP_W, TOP_H), (0, TOP_H),
               (TOP_W / 2, 0), (TOP_W / 2, TOP_H)]


# colori standard delle biglie (BGR); 9-15 = stessi colori a righe
BALL_BGR = {"1": (0, 215, 255), "2": (200, 0, 0), "3": (0, 0, 220), "4": (150, 30, 140),
            "5": (0, 140, 255), "6": (0, 150, 0), "7": (40, 40, 140), "8": (20, 20, 20),
            "0": (255, 255, 255), "?": (130, 130, 130)}


def base_color(label):
    if label in BALL_BGR:
        return BALL_BGR[label]
    n = int(label)
    return BALL_BGR[str(n - 8)] if n > 8 else BALL_BGR.get(label, (130, 130, 130))


def draw_2d_map(balls_top, segments, R_top, capture, result_txt, out, cue_defl=None):
    """Mappa 2D schematica (vista dall'alto rettificata) con tavolo, buche,
    palle numerate e la catena della predizione."""
    M = 70
    W, H = int(TOP_W), int(TOP_H)
    canvas = np.full((H + 2 * M, W + 2 * M, 3), (70, 70, 70), np.uint8)
    cv2.rectangle(canvas, (M, M), (M + W, M + H), (150, 110, 40), -1)   # panno
    cv2.rectangle(canvas, (M, M), (M + W, M + H), (40, 40, 40), 4)      # sponde
    for q in POCKETS_TOP:                                               # buche
        cv2.circle(canvas, (int(q[0]) + M, int(q[1]) + M), 20, (25, 25, 25), -1)

    palette = [(0, 200, 0), (255, 200, 0), (255, 0, 200), (0, 200, 255)]
    for k, (seg, cur) in enumerate(segments):                          # catena
        color = (0, 0, 255) if cur is None else palette[(k - 1) % len(palette)]
        pts = [(int(p[0]) + M, int(p[1]) + M) for p in seg]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, color, 2)
    if cue_defl is not None:                                            # battente dopo urto
        pts = [(int(p[0]) + M, int(p[1]) + M) for p in cue_defl]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, (200, 200, 200), 2)

    r = max(10, int(R_top))
    for ctop, label in balls_top:                                      # palle
        c = (int(ctop[0]) + M, int(ctop[1]) + M)
        cv2.circle(canvas, c, r, base_color(label), -1)
        if label not in ("0", "?") and int(label) > 8:                 # righe -> anello bianco
            cv2.circle(canvas, c, r, (255, 255, 255), 2)
        cv2.circle(canvas, c, r, (0, 0, 0), 1)
        cv2.putText(canvas, label, (c[0] - 7, c[1] - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(canvas, result_txt, (M, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (0, 255, 0) if result_txt.startswith("IN") else (0, 0, 255), 3)
    cv2.imwrite(out, canvas)


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


def first_ball_hit(P, d, R, balls, skip, t_max):
    """Prima palla (indice non in skip) colpita lungo P+t d entro t_max.
    Contatto quando i centri distano 2R."""
    best_t, best_i = None, None
    for i, C in enumerate(balls):
        if i in skip:
            continue
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


def propagate(P, d, R, balls, skip, capture, max_bounces=0):
    """Un ANELLO della catena: la palla avanza fino al primo evento.
    max_bounces=0 -> si ferma alla prima sponda (retta pulita).
    max_bounces>0 -> rimbalza sulle sponde (banchi) fino a incontrare palla/buca.
    Ritorna (path, evento): ('ball', j, contact, struck_dir) | ('pocket',) | ('none',)."""
    pockets = [np.array(q, np.float64) for q in POCKETS_TOP]
    P = P.astype(np.float64); d = d / (np.linalg.norm(d) + 1e-9)
    path = [P.copy()]
    for _ in range(max_bounces + 1):
        t_wall, ax = nearest_wall(P, d, R)
        if t_wall is None:
            break
        t_ball, j = first_ball_hit(P, d, R, balls, skip, t_wall)
        if t_ball is not None:                        # colpisce una palla
            contact = P + d * t_ball
            path.append(contact)
            sdir = balls[j] - contact
            return path, ("ball", j, contact, sdir / (np.linalg.norm(sdir) + 1e-9))
        hit = P + d * t_wall                           # arriva alla sponda
        path.append(hit)
        if min(np.linalg.norm(hit - q) for q in pockets) <= capture:
            return path, ("pocket",)
        d = np.array([-d[0], d[1]]) if ax == "x" else np.array([d[0], -d[1]])
        P = hit
    return path, ("none",)


def simulate_chain(P0, d0, R, balls, capture, max_depth=3, max_bounces=0):
    """Catena: battente -> palla A -> (A colpisce B -> B ...) fino a max_depth.
    Ritorna (segmenti, potted_idx) dove segmenti = lista di (path, ball_idx|None)
    e potted_idx = indice palla imbucata (None = nessuna, -1 = battente/scratch)."""
    segments = []
    skip = set()
    P, d, cur = P0, d0, None          # cur=None -> battente
    potted_idx = None
    for _ in range(max_depth + 1):
        seg, ev = propagate(P, d, R, balls, skip, capture, max_bounces)
        segments.append((seg, cur))
        if ev[0] == "pocket":
            potted_idx = cur if cur is not None else -1
            break
        if ev[0] == "ball":
            j = ev[1]
            skip.add(j)
            P, d, cur = balls[j], ev[3], j
            continue
        break                          # 'none' -> la palla si ferma, catena finita
    return segments, potted_idx


def draw_path(ov, path_top, Hinv, color, thick=3):
    pi = cv2.perspectiveTransform(np.array([path_top], np.float32), Hinv)[0]
    for a, b in zip(pi[:-1], pi[1:]):
        cv2.line(ov, tuple(a.astype(int)), tuple(b.astype(int)), color, thick)


def run_prediction(frame, model, max_bounces=0):
    """Esegue tutta la predizione su un frame. Ritorna un dict con esito e dati
    per il disegno, oppure None se manca battente/stecca."""
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
        return None
    cue_img, R_img = cue["center"], cue["radius"]
    _, shot_img = detect_cue(frame, model, ball=cue_img)
    if shot_img is None:
        return None

    obj = [b for b in balls if b is not cue]
    obj_top = [to_top(b["center"]) for b in obj]
    P0 = to_top(cue_img)
    d0 = to_top(cue_img + shot_img * 200) - P0

    # RAGGIO robusto: mediana del raggio di TUTTE le palle (non solo la battente)
    def rtop(b):
        return float(np.linalg.norm(to_top(b["center"] + np.array([b["radius"], 0], np.float32))
                                    - to_top(b["center"])))
    R_top = float(np.median([rtop(b) for b in balls])) if balls else \
        float(np.linalg.norm(to_top(cue_img + np.array([R_img, 0], np.float32)) - P0))
    capture = 2.2 * R_top

    segments, potted_idx = simulate_chain(P0, d0, R_top, obj_top, capture,
                                          max_depth=3, max_bounces=max_bounces)
    chain = [obj[cur]["label"] for _, cur in segments if cur is not None]
    if potted_idx is None:
        result, potted_label = "OUT", None
    elif potted_idx == -1:
        result, potted_label = "SCRATCH", None
    else:
        result, potted_label = "IN", obj[potted_idx]["label"]

    # DEVIAZIONE della battente dopo il PRIMO urto (regola 90°: la battente esce
    # lungo la TANGENTE, perpendicolare alla congiungente dei centri).
    cue_defl, cue_scratch = None, False
    if len(segments) >= 2 and segments[0][1] is None and segments[1][1] is not None:
        cue_seg = segments[0][0]
        contact = cue_seg[-1]
        d_in = contact - cue_seg[-2]
        d_in = d_in / (np.linalg.norm(d_in) + 1e-9)
        n = obj_top[segments[1][1]] - contact
        n = n / (np.linalg.norm(n) + 1e-9)               # congiungente centri
        tang = d_in - float(d_in @ n) * n                # componente perpendicolare
        if np.linalg.norm(tang) > 1e-6:
            tang = tang / np.linalg.norm(tang)
            seg, ev = propagate(contact, tang, R_top, obj_top,
                                {segments[1][1]}, capture, max_bounces)
            cue_defl = seg
            cue_scratch = (ev[0] == "pocket")

    return dict(balls=balls, obj=obj, cue=cue, cue_img=cue_img, R_img=R_img,
                segments=segments, potted_idx=potted_idx, chain=chain,
                result=result, potted_label=potted_label,
                cue_defl=cue_defl, cue_scratch=cue_scratch,
                Hinv=Hinv, to_top=to_top, R_top=R_top, capture=capture)


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "datasets/videos/video3.mp4"
    frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    max_bounces = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # >0 = permetti banchi (prova)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("frame non letto")

    model = YOLO(WEIGHTS)
    pred = run_prediction(frame, model, max_bounces)
    if pred is None:
        raise SystemExit("battente o stecca non trovata")
    balls, obj, cue_img, R_img = pred["balls"], pred["obj"], pred["cue_img"], pred["R_img"]
    segments, potted_idx = pred["segments"], pred["potted_idx"]
    Hinv, to_top, R_top, capture = pred["Hinv"], pred["to_top"], pred["R_top"], pred["capture"]
    cue_defl, cue_scratch = pred["cue_defl"], pred["cue_scratch"]

    ov = frame.copy()
    for q in POCKETS_TOP:
        c = cv2.perspectiveTransform(np.array([[q]], np.float32), Hinv)[0, 0]
        cv2.circle(ov, tuple(c.astype(int)), int(capture), (0, 255, 255), 1)

    # numero su OGNI palla
    for b in balls:
        col = (255, 255, 255) if b["label"] == "0" else (0, 255, 255)
        draw_ball_label(ov, b["center"], b["label"], col)

    # disegno la catena: battente=rosso, poi ogni anello un colore diverso
    palette = [(0, 200, 0), (255, 200, 0), (255, 0, 200), (0, 200, 255)]
    chain = []
    for k, (seg, cur) in enumerate(segments):
        color = (0, 0, 255) if cur is None else palette[(k - 1) % len(palette)]
        draw_path(ov, seg, Hinv, color, 3)
        if cur is not None:
            chain.append(obj[cur]["label"])
    if cue_defl is not None:                       # battente dopo l'urto (tangente, 90°)
        draw_path(ov, cue_defl, Hinv, (200, 200, 200), 2)
    cv2.circle(ov, tuple(cue_img.astype(int)), int(R_img), (255, 255, 255), 2)
    print(f"Catena: battente -> " + " -> ".join(chain) if chain else "Nessuna palla colpita")

    if potted_idx is None:
        result, txt = "OUT", "OUT"
    elif potted_idx == -1:
        result, txt = "SCRATCH", "SCRATCH (battente in buca)"
    else:
        result = "IN"
        lbl = obj[potted_idx]["label"]
        txt = f"IN: palla {lbl}"
        cv2.circle(ov, tuple(obj[potted_idx]["center"].astype(int)), int(R_img) + 4, (0, 200, 0), 3)
        print(f"Imbucata: palla {lbl}")
    if cue_scratch:
        txt += " + SCRATCH"
        print("ATTENZIONE: la battente rischia la buca dopo l'urto (scratch)")
    cv2.putText(ov, txt, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0, 255, 0) if result == "IN" else (0, 0, 255), 5)
    name = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
    out = f"output/predict_{name}_f{frame_idx}.png"
    cv2.imwrite(out, ov)

    # mappa 2D schematica (vista dall'alto)
    balls_top = [(to_top(b["center"]), b["label"]) for b in balls]
    map_out = f"output/map_{name}_f{frame_idx}.png"
    draw_2d_map(balls_top, segments, R_top, capture, txt, map_out, cue_defl)
    print(f"Esito: {txt}  |  salvato {out}  +  {map_out}")


if __name__ == "__main__":
    main()
