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
Doppio output: overlay sull'**immagine** (`predict_<video>_f<frame>.png`) **e mappa
2D** schematica dall'alto (`map_<video>_f<frame>.png`, colori standard, righe con
anello bianco). Rifiniture: `R`/`CAPTURE` dalla **mediana** del raggio di tutte le
palle; **deviazione battente** dopo l'urto (tangente, regola 90°, linea grigia) +
rilevamento **scratch**. Assunzione onesta: geometria = *dove*, non *quanto* (niente energia).

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

### Validazione (`validate.py`) — Estensione C
Confronta predizione vs realtà nei frame successivi: predico sul frame del tiro,
poi su un frame DOPO (a palle ferme) le palle SPARITE dal gioco aperto = imbucate.
`run_prediction(frame, model)` è la funzione riusabile estratta da `predict.py`.
Onesto: se il conteggio palle *aumenta* è rumore → inconcludente. Esiti: video2
IN palla 4 ✅ confermato; video4 era un **banco** che imbuca la 7 (default lo manca,
la modalità banchi lo prende); video3 inconcludente per rumore.

### Omografia obliqua (`detect_oblique.py`) — Estensione B
Demo su V2 obliquo: 4 angoli **a mano** (auto-detection non regge sul grigio) →
omografia → raddrizzo; validazione per auto-consistenza (palle → circolari).
Su immagine PULITA (`out-1-101`, tavolo intero) con angoli marcati bene (griglia;
la buca MURREY è la centrale non un angolo) e aspetto tarato (~2.7:1): **funziona**,
palle circolari (aspetto ~1.04). Residuo onesto: CV diametri ~29% = **distorsione
lente** (grandangolo, servirebbe calibrazione camera); 2.7:1 vs 2:1 = angoli non
perfetti al pixel. Niente GT accoppiata → errore per auto-consistenza.

## Stato
Nucleo completo end-to-end (tavolo ✅, omografia ✅, buche ✅, palle ✅, stecca ✅,
predizione ✅) + Estensione A (catena, banchi opzionali) + Estensione B (omografia
obliqua, con limiti) + Estensione C (validazione). Gira su materiale reale (video2/3/4).

## Prossimi possibili passi (opzionali)
- **Calibrazione camera** per migliorare l'Estensione B (rimuovere la distorsione
  della lente prima del raddrizzamento) — serve una scacchiera di calibrazione.
- Detection automatica del tavolo su panno **grigio/obliquo** (V2), oggi manuale.
- Validazione C più robusta all'identità rumorosa (es. matching per posizione).
