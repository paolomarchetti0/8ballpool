"""
Detection palle: YOLO trova le posizioni, il COLORE decide l'identita'.

YOLO (best.pt) e' bravo a LOCALIZZARE le palle ma inaffidabile sul NUMERO
(le palle del nostro video sono renderizzate, diverse da quelle d'allenamento).
Qui uso YOLO solo per i box, poi classifico ogni palla dai suoi pixel:
  - quasi tutta bianca, senza colore     -> battente 0
  - quasi tutta nera                      -> 8
  - altrimenti: tinta dominante (hue) -> famiglia colore 1..7,
    e piena/righe (frazione di bianco) -> X (piena) o X+8 (righe).

Schema colori standard:
  1 giallo 2 blu 3 rosso 4 viola 5 arancione 6 verde 7 bordeaux 8 nero
  9..15 = righe degli stessi colori (9 giallo ... 15 bordeaux), 0 bianca.
"""
import sys
import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = "runs/detect/full/weights/best.pt"
NAMES = ['0', '1', '10', '11', '12', '13', '14', '15',
         '2', '3', '4', '5', '6', '7', '8', '9']


def hue_to_base(hue, val):
    """hue OpenCV (0-179) + luminosita' -> numero base 1..7 (colore pieno).
    Confini calibrati sui colori misurati nel nostro video."""
    if hue < 8 or hue >= 160:          # rosso / bordeaux
        return 7 if val < 130 else 3   # bordeaux (scuro) vs rosso (chiaro)
    if hue < 20:   return 5            # arancione
    if hue < 34:   return 1            # giallo
    if hue < 85:   return 6            # verde
    if hue < 123:  return 2            # blu
    return 4                           # viola


def ball_features(crop_bgr):
    """Misure di colore su una palla ritagliata."""
    h, w = crop_bgr.shape[:2]
    cy, cx = h // 2, w // 2
    r = max(2, int(0.42 * min(h, w)))
    Y, X = np.ogrid[:h, :w]
    d2 = (X - cx) ** 2 + (Y - cy) ** 2
    disk = d2 <= r * r
    ring = (d2 >= (0.62 * r) ** 2) & (d2 <= (0.95 * r) ** 2)   # anello di bordo

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (V > 160) & (S < 45)
    dark = (V < 95)
    colored = (S > 80) & (V >= 95)
    tot = max(1, int(disk.sum()))
    rtot = max(1, int(ring.sum()))
    f = dict(
        wf=(white & disk).sum() / tot,
        df=(dark & disk).sum() / tot,
        cf=(colored & disk).sum() / tot,
        border_white=(white & ring).sum() / rtot,  # righe: bianco alle calotte
    )
    cm = colored & disk
    if cm.sum() >= 10:
        f["hue"] = float(np.median(H[cm]))
        f["val"] = float(np.median(V[cm]))
    else:                      # poco colore: allargo per catturare tinte deboli
        cm2 = (S > 45) & (V >= 95) & disk
        f["hue"] = float(np.median(H[cm2])) if cm2.sum() else -1.0
        f["val"] = float(np.median(V[cm2])) if cm2.sum() else 0.0
    return f


def label_from_features(f, allow_cue=True):
    """Etichetta una palla. allow_cue=False forza un colore (per demanare i doppioni bianchi)."""
    if allow_cue and f["df"] > 0.5 and f["cf"] < 0.25:
        return "8"
    if allow_cue and f["wf"] > 0.5 and f["cf"] < 0.20:
        return "0"
    if f["hue"] < 0:
        return "8" if f["df"] > f["wf"] else "0"
    base = hue_to_base(f["hue"], f["val"])
    # righe vs piena:
    #  - caso normale: POCA area colorata E bianco alle calotte del bordo;
    #  - VINCOLO: una PIENA (1-8) non ha bianco. Se c'e' MOLTO bianco (wf alto)
    #    non puo' essere piena -> forzo spezzata (o e' la battente, gia' gestita).
    stripe = (f["cf"] < 0.60 and f["border_white"] > 0.15) or f["wf"] > 0.42
    return str(base + 8 if stripe else base)


def classify_ball(crop_bgr):
    if crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
        return "?", {}
    f = ball_features(crop_bgr)
    return label_from_features(f), {k: round(v, 2) for k, v in f.items()}


def detect(image_bgr, conf=0.2, model=None, verbose=False):
    """Rileva e classifica le palle. Accetta path o immagine BGR."""
    model = model or YOLO(WEIGHTS)
    r = model.predict(image_bgr, conf=conf, augment=True, verbose=False)[0]
    img = r.orig_img.copy()

    # box + confidenza, ordinati per confidenza
    raw = [(b.xyxy[0].int().tolist(), float(b.conf[0])) for b in r.boxes]
    raw.sort(key=lambda t: t[1], reverse=True)

    # dedup: scarto box il cui centro cade dentro un box gia' tenuto (TTA duplica)
    kept = []
    for (x1, y1, x2, y2), cf in raw:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dup = any(kx1 <= cx <= kx2 and ky1 <= cy <= ky2
                  for (kx1, ky1, kx2, ky2) in kept)
        if not dup:
            kept.append((x1, y1, x2, y2))

    feats = [(box, ball_features(r.orig_img[box[1]:box[3], box[0]:box[2]]))
             for box in kept]

    labels = [label_from_features(f) for _, f in feats]

    # Regola UNA SOLA battente: se piu' palle risultano '0', tengo come battente
    # solo la piu' bianca; le altre le rivaluto forzando un colore.
    cue_idx = [i for i, l in enumerate(labels) if l == "0"]
    if len(cue_idx) > 1:
        keep = max(cue_idx, key=lambda i: feats[i][1]["wf"])
        for i in cue_idx:
            if i != keep:
                labels[i] = label_from_features(feats[i][1], allow_cue=False)

    balls = []
    for (box, f), lab in zip(feats, labels):
        balls.append((box, lab, f))
        if verbose:
            print(f"  {box} -> {lab:>2}  wf={f['wf']:.2f} cf={f['cf']:.2f} "
                  f"bw={f['border_white']:.2f} hue={f['hue']:.0f}")
    return img, balls


def identify_balls(frame, model, conf=0.25):
    """Pipeline identita' completa: YOLO + dedup + una-battente + colore su
    declassate + unicita' + recupero '?'. Ritorna lista di dict:
    {box, center, radius, label, conf}."""
    r = model.predict(frame, conf=conf, augment=True, verbose=False)[0]
    dets = sorted([(b.xyxy[0].int().tolist(), int(b.cls[0]), float(b.conf[0]))
                   for b in r.boxes], key=lambda t: t[2], reverse=True)
    kept = []
    for (x1, y1, x2, y2), c, cf in dets:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(kx1 <= cx <= kx2 and ky1 <= cy <= ky2
               for (kx1, ky1, kx2, ky2), _, _ in kept):
            continue
        kept.append(((x1, y1, x2, y2), c, cf))

    labels = [NAMES[c] for _, c, _ in kept]
    uconf = [cf for _, _, cf in kept]
    cue_idx = [i for i, l in enumerate(labels) if l == "0"]
    if len(cue_idx) > 1:
        feats = {i: ball_features(frame[kept[i][0][1]:kept[i][0][3],
                                        kept[i][0][0]:kept[i][0][2]]) for i in cue_idx}
        keep = max(cue_idx, key=lambda i: feats[i]["wf"])
        for i in cue_idx:
            if i != keep:
                labels[i] = label_from_features(feats[i], allow_cue=False)
                uconf[i] = 0.0
    seen = set()
    for i in sorted(range(len(kept)), key=lambda i: uconf[i], reverse=True):
        if labels[i] in ("0", "?"):
            continue
        if labels[i] in seen:
            labels[i] = "?"
        else:
            seen.add(labels[i])
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
                    break

    balls = []
    for ((x1, y1, x2, y2), c, cf), lab in zip(kept, labels):
        balls.append(dict(box=(x1, y1, x2, y2),
                          center=np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32),
                          radius=(x2 - x1 + y2 - y1) / 4.0, label=lab, conf=cf))
    return balls


def draw(img, balls, out):
    for (x1, y1, x2, y2), label, _ in balls:
        color = (0, 255, 0) if label == "0" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    cv2.imwrite(out, img)


if __name__ == "__main__":
    frames = sys.argv[1:] or ["output/sample_00_frame00072.png",
                              "output/sample_02_frame00218.png"]
    for f in frames:
        print(f)
        img, balls = detect(f)
        out = f.replace("output/", "output/balls_")
        draw(img, balls, out)
        print(f"  -> {len(balls)} palle, salvato {out}\n")
