# CLAUDE.md — Pool Shot Predictor

Guida sintetica per lavorare su questa repo: obiettivo, decisioni prese e come
girare le cose. Documentazione estesa in `DOCUMENTATION.md`.

## Obiettivo
Predittore **geometrico** di tiri al biliardo da **un frame "prima del colpo"**.
Predizione istantanea: **NO fisica/energia/spin/velocità, NO frame successivi**.
Dove possibile si usa CV **classica**; le palle con **YOLO addestrato da noi**.

## Modo di lavoro (importante)
- **Un pezzo alla volta**: fermarsi, spiegare, aspettare conferma. Non aprire più
  blocchi insieme.
- Codice **semplice e didattico** (va spiegato all'esame). Segnalare **assunzioni e
  punti fragili**. Preferire approcci **data-driven** a costanti "a occhio".
- Ispezionare il materiale prima di scrivere codice.

## Ambiente
- Windows, Python 3.11, GPU NVIDIA (RTX 3060 Ti, CUDA 12.4). Venv `.venv`.
- PyTorch a parte: `pip install torch torchvision --index-url .../whl/cu124`, poi
  `opencv-python numpy ultralytics`. `data.yaml` va scritto in **UTF-8** (path con "à").

## Decisioni chiave

### Tavolo (`detect_table.py`)
Soglia **HSV auto-stimata** dal centro (si adatta a ciano/blu) → maschera panno →
`minAreaRect` → 4 angoli → omografia verso rettangolo **1000×500** (2:1). Il bordo
del panno = naso delle sponde = linea dei rimbalzi.

### Buche (`detect_pockets.py`)
Data-driven dalla maschera: **chiusura morfologica** riempie le insenature (le
buche), `chiuso − panno` = zone scure. **Robusto all'occlusione**: scarto blob
troppo grandi (corpo del giocatore) e aggancio i rimanenti alle 6 posizioni attese
(4 angoli + 2 metà lati lunghi); buca occlusa → fallback geometrico. Scartati:
offset "a occhio", convex hull, rettangolo−panno.
Imbucata = **varco nella sponda**; nel codice approssimato con `CAPTURE` (≈2.2·R):
cerchio piccolo per evitare falsi positivi (non deve coprire il verde).

### Palle (`detect_balls.py`) — decisione importante
YOLO **localizza** bene; sui **numeri** è inaffidabile su domini diversi (es. video
ciano). Quindi: **YOLO per le posizioni, COLORE per l'identità**. Non riaddestrare:
il dominio è vicino, sono palle vere (non render), lo scarto è piccolo e mirato.
Pipeline `identify_balls`: dedup box → **una sola battente** (la più bianca) →
**bianco⇒non piena** → **unicità** (numeri unici; duplicati→`?`) → **recupero `?`**
via ipotesi YOLO a conf bassa (NMS per-classe) tra i numeri liberi.
Nota: per la geometria bastano **posizioni + battente**; i numeri esatti delle altre
palle non sono critici (ostacoli anonimi). Le occluse restano `?`.

### Stecca (`detect_cue.py`)
ROI interno tavolo → Canny + `HoughLinesP` → segmenti lunghi che **passano vicino
alla battente** → fit retta (asse) **campionando lungo tutta la lunghezza** (più
preciso). Verso: `bianca − baricentro_stecca`. La **precisione dell'angolo** è il
punto fragile n.1 (nei tiri frontali si amplifica).

### Predizione (`predict.py`)
Geometria in **vista rettificata**, poi riproiettata. Ghost-ball: la palla colpita
parte lungo la **congiungente dei centri** (contatto a 2R). **Catena a profondità
limitata** (`max_depth=3`): battente → prima palla → palla successiva… Esito
`IN: palla X` / `OUT` / `SCRATCH`. **Banchi opzionali** (3° argomento `max_bounces`,
default 0 = retta pulita). Numero disegnato su ogni palla.
Assunzione onesta: geometria = *dove*, non *quanto* (niente energia).

### Dataset / training
Due dataset (V2 grigio-obliquo, V3 blu-dall'alto), 16 classi, `names` string-sort
(indice `i`→`names[i]`, es. 2→"10"). Sono frame di **soli 6 video** → lo split
casuale di Roboflow ha **leakage**. → Dataset unito `datasets/merged/` con **split
PER VIDEO** (`build_merged.py`): valid = `V3/video-3`. Modello finale
`runs/detect/full/weights/best.pt` (YOLOv8s, 80 epoch, mAP50 0.877 onesto).

## Comandi
```bash
python scripts/run_core.py "datasets/videos/video3.mp4" 0        # tavolo+buche+palle
python scripts/detect_cue.py "datasets/videos/video3.mp4" 0      # stecca
python scripts/predict.py "datasets/videos/video2.mp4" 69        # predizione (catena)
python scripts/predict.py "datasets/videos/video4.mp4" 0 6       # con banchi (prova)
python scripts/build_merged.py && python scripts/train.py full   # dataset + training
```
Usare **frame fermi** (motion≈0) per la detection; per la stecca un frame con la
**stecca in mira**. Output in `output/predict_<video>_f<frame>.png`.

## Stato
Nucleo completo end-to-end (tavolo ✅, omografia ✅, buche ✅, palle ✅, stecca ✅,
predizione ✅) + Estensione A (catena di collisioni, banchi opzionali). Gira su
materiale reale (video2/3/4).

## Prossimi possibili passi
- Validare gli esiti IN/OUT sui **frame successivi** (Estensione C).
- Legare `CAPTURE`/`R` al **raggio palla vero** (casi borderline).
- Deviazione della battente dopo l'urto (tangente / regola 90°).
- Omografia obliqua validata (Estensione B).
