# Pool Shot Predictor — Documentazione

Progetto d'esame di Computer Vision (A.A. 2025/2026).
Autori: Lorenzo Zanda, Davide Vittucci, Paolo Marchetti.

Sistema che, dato **un frame "prima del colpo"** (palla ferma, stecca in mira),
predice **geometricamente** il percorso della battente, le collisioni, i rimbalzi
e se una palla finisce in buca. La predizione è **istantanea e geometrica**: NON
usa i frame successivi per predire, NON modella fisica/energia/spin/velocità.

---

## 1. Pipeline (visione d'insieme)

```
frame del tiro
   │
   ├── TAVOLO (classico)      → maschera HSV del panno → 4 angoli → OMOGRAFIA (vista dall'alto)
   ├── BUCHE (classico)       → insenature della maschera (chiusura morfologica), robuste all'occlusione
   ├── PALLE (YOLO nostro)    → posizioni + identità (YOLO + regola a colore + vincoli di dominio)
   └── STECCA (classico Hough)→ asse della stecca + verso del colpo
   │
   ▼
PREDIZIONE (in vista dall'alto rettificata, poi riproiettata):
   battente → (rimbalzi sponde) → prima palla → ghost-ball → catena di collisioni → esito IN/OUT + quale palla
```

I tre blocchi di detection (tavolo, palle, stecca) sono **indipendenti**; la
predizione li usa tutti insieme. La geometria si fa sulla **vista dall'alto
rettificata** (riflessioni pulite = rettangolo asse-allineato), poi si
**riproietta** sull'immagine per il disegno.

---

## 2. Ambiente

- Windows, Python 3.11, GPU NVIDIA (sviluppato su RTX 3060 Ti 8 GB, CUDA 12.4).
- Venv `.venv`. Dipendenze in `requirements.txt`.
- **PyTorch va installato a parte** con l'index CUDA:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install opencv-python numpy ultralytics
  ```
- Verifica GPU: `torch.cuda.is_available()` deve dare `True`.

---

## 3. Struttura della repository

```
8ball-pool-predictor/
├── datasets/
│   ├── videos/                         # video di input (pool 1, video2..5)
│   ├── Pool Ball Detection V2.yolov8/   # dataset sorgente (grigio, obliquo)
│   ├── Pool Ball Detection V3.yolov8/   # dataset sorgente (blu, dall'alto)
│   └── merged/                          # dataset unito 16 classi, split PER VIDEO (generato)
├── runs/detect/full/weights/best.pt     # modello YOLO addestrato (generato)
├── scripts/                             # tutto il codice della pipeline
├── src/                                 # scheletro (config, utils, seed)
├── output/                              # overlay e immagini diagnostiche (generato)
├── requirements.txt
├── DOCUMENTATION.md
└── CLAUDE.md                            # sintesi delle decisioni
```

`src/` è lo scheletro iniziale (`config.py` con SEED/percorsi, `utils.py` con
`set_seed`, `main.py`). La pipeline vera è tutta in `scripts/`.

---

## 4. Dati e training YOLO

### Dataset sorgente
Due dataset YOLOv8 (stesso autore, Roboflow), già a 16 classi (0-15, niente
classe "rack"):
- **V2**: ~1042 img, panno grigio, inquadratura obliqua (sala giochi).
- **V3**: ~995 img, panno blu, dall'alto (match professionale) — più simile ai video.

L'ordine `names` nei due `data.yaml` è identico (string-sort: `0,1,10,11,…,15,2,…,9`),
quindi **l'indice YOLO `i` → etichetta `names[i]`** (attenzione: indice 2 = palla "10").

### QA prima del training (`qa_datasets.py`, `qa_labels.py`, `analyze_videos.py`)
- **Qualità label**: disegno box+numero su un campione → label corrette in entrambi.
- **Bilanciamento**: combinato ~3.5×; le palle 11-15 sono le più deboli (colpa di V3).
- **Leakage**: i dataset sono frame di **soli 6 video** (2 in V2, 4 in V3);
  Roboflow ha splittato a caso → ~55% dei frame vicini finisce in split diversi
  → le metriche val/test "ufficiali" sono **gonfiate**.

### Dataset unito con split PER VIDEO (`build_merged.py`)
Per avere metriche oneste, rifacciamo lo split **per video** (interi video in
train o in valid, mai spezzati): niente leakage.
- **valid** = `V3/video-3` (dall'alto, mai visto in training).
- **train** = gli altri 5 video → 1696 train / 341 valid.
- Caveat: le classi 11,12,13,15 non sono in valid (assenti in video-3); restano in
  train dove servono. Il vero giudice resta **il nostro video**.
- **Nota tecnica**: scrivere `data.yaml` in **UTF-8** (il percorso contiene "à").

### Training (`train.py`)
Transfer learning da YOLOv8. `python scripts/train.py` = prova (YOLOv8n, 3 epoch,
verifica GPU/pipeline); `python scripts/train.py full` = completo (YOLOv8s, 80 epoch).
- Modello finale: `runs/detect/full/weights/best.pt`, **mAP50 = 0.877** sullo split
  onesto per video.

---

## 5. Descrizione degli script

### `inspect_video.py`
Apre un video, stampa proprietà (risoluzione, fps, durata) e salva alcuni frame
campione. Serve a scegliere il frame del tiro.

### `detect_table.py` — TAVOLO + OMOGRAFIA
- `cloth_mask(frame)`: soglia **HSV auto-stimata** dal centro del frame (si adatta
  a panno ciano/blu). Morfologia per tappare i buchi (palle/buche). Restituisce la
  maschera del panno + colore mediano.
- `order_corners`: ordina 4 punti in alto-sx, alto-dx, basso-dx, basso-sx.
- `main`: `minAreaRect` sul blob più grande → 4 angoli → `findHomography` verso un
  rettangolo canonico **1000×500** (proporzioni ~2:1). Salva overlay + vista dall'alto.
- Nota: il **bordo del panno = naso delle sponde**, cioè la linea su cui le palle
  rimbalzano → la maschera serve direttamente alla fisica dei rimbalzi.

### `detect_pockets.py` — BUCHE (data-driven, robuste all'occlusione)
Le buche sono le **insenature concave** del bordo del panno.
- Riempio il contorno del panno (spariscono i buchi interni = palle).
- **Chiusura morfologica** (kernel ~121px): riempie solo le insenature (le buche),
  non aggiunge bande lungo i lati diritti.
- `panno_chiuso − panno` = le zone scure delle buche.
- `find_pockets(mask, corners)`: scarta i blob **troppo grandi** (`MAX_POCKET_AREA`;
  un giocatore appoggiato fa una baia ~20000, 10× una buca ~1500) e **aggancia** i
  blob rimasti alle 6 posizioni attese (4 angoli + 2 metà lati lunghi). Buca occlusa
  senza blob vicino → **fallback geometrico**.
- Metodi scartati (documentati nel codice): offset "a occhio", convex hull, rettangolo−panno.

### `detect_balls.py` — PALLE (YOLO posizioni + COLORE identità)
Lo YOLO **localizza** bene le palle ma sui **numeri** può sbagliare quando il
dominio è diverso (es. il video ciano `pool 1`): panno + luce spostano alcuni colori.
Soluzione (senza riaddestrare): YOLO per le posizioni, **identità dal colore**.
- `ball_features`: misura su un disco della palla — `wf` (bianco), `df` (scuro),
  `cf` (area colorata), `border_white`, hue/val mediani.
- `label_from_features`: **8** = molto scura (tinta ignorata); **battente 0** = molto
  bianca senza colore; colorate → tinta→colore base 1..7 (blu<123<viola; rosso/bordeaux
  per val alto/basso) e **piena vs righe** (`cf<0.60 AND border_white>0.15`, oppure
  `wf>0.42` → una piena non ha banda bianca).
- `identify_balls(frame, model)`: pipeline completa =
  dedup box (TTA duplica) → **una sola battente** (tra le "0" tengo la più bianca,
  le altre rietichettate col colore) → **unicità** (ogni numero una volta; duplicati
  → `?`; una declassata ha confidenza 0) → **recupero `?`** (la NMS di YOLO è
  per-classe: a conf bassa la palla riappare sotto altre classi → prendo la migliore
  classe libera). Restituisce `{box, center, radius, label, conf}` per ogni palla.
- Validato su frame fermi: **10/11 corrette e stabili**, battente affidabile.

### `detect_cue.py` — STECCA (classico Hough)
- `cue_ball_center`: la battente (classe "0" YOLO più confidente).
- `detect_cue(frame, model, ball)`: ROI = interno tavolo (maschera erosa); Canny +
  `HoughLinesP`; tengo i segmenti **lunghi che passano vicino alla battente** (le
  sponde e le altre linee no); **campiono punti lungo tutta la lunghezza** dei
  segmenti (i lunghi pesano di più → angolo più preciso) e fitto la retta (asse).
  **Verso**: la stecca sta dietro la bianca → direzione = `bianca − baricentro_stecca`.

### `predict.py` — PREDIZIONE geometrica (nucleo + Estensione A)
Tutto in **vista dall'alto rettificata** (omografia dai 4 angoli), poi riproiettato.
- `nearest_wall`: prima sponda incontrata nel rettangolo interno `[R, W-R]×[R, H-R]`
  (il **centro** palla rimbalza a distanza R dalla sponda).
- `first_ball_hit`: prima palla colpita (centri a distanza 2R), escludendo le palle
  già mosse (`skip`).
- `propagate`: un **anello** della catena. `max_bounces=0` → retta pulita fino al
  primo evento; `max_bounces>0` → **banchi** (rimbalzi) fino a incontrare palla/buca.
- `simulate_chain`: **catena a profondità limitata** (`max_depth=3`): battente → prima
  palla → ghost-ball (la colpita parte lungo la **congiungente dei centri**) →
  eventuale palla successiva → … Imbucata quando un anello finisce su una **buca**.
- **Rifiniture**: `R_top` (e `CAPTURE=2.2·R`) dalla **mediana del raggio di tutte le
  palle** (più robusto del solo box battente); **deviazione della battente** dopo il
  primo urto lungo la **tangente** (regola dei 90°, linea grigia) con rilevamento
  dello **scratch** (battente in buca → esito `+ SCRATCH`).
- `main`: rileva tavolo+buche+battente+stecca, simula, disegna la catena (battente
  rossa, anelli successivi con colori diversi), **numero su ogni palla**, e l'esito
  **`IN: palla X` / `OUT` / `SCRATCH`**.
- `draw_2d_map`: **mappa 2D schematica** (vista dall'alto rettificata) con tavolo,
  buche, palle numerate coi **colori standard** (righe 9-15 con anello bianco) e la
  catena. Doppio output: `output/predict_<video>_f<frame>.png` (immagine) +
  `output/map_<video>_f<frame>.png` (mappa 2D).

### `run_core.py`
Driver che su un frame mostra **tavolo + buche + palle (identità YOLO native + vincoli)**
in un unico overlay, senza la predizione. Utile per verificare la detection.

### `validate.py` — VALIDAZIONE esito (Estensione C)
Confronta la predizione col risultato reale nei frame successivi (usati **solo**
per validare): predice sul frame del tiro, poi su un frame **dopo** (a palle ferme,
auto-scelto come ultimo frame fermo) rileva le palle rimaste in **gioco aperto**
(lontane dalle buche); le palle **sparite** = imbucate davvero. Verdetto COERENTE/
NON coerente. Onesto sui limiti: se il conteggio *aumenta* è rumore di identità →
validazione inconcludente. `run_prediction(frame, model)` è la funzione riusabile
estratta da `predict.py`.
Risultati: video2 (IN palla 4) confermato; video4 (banco che imbuca la 7) preso
dalla modalità banchi; video3 inconcludente per rumore.

### `detect_oblique.py` — OMOGRAFIA OBLIQUA (Estensione B)
Demo su immagine obliqua reale (dataset V2, panno grigio): 4 angoli del tavolo
**marcati a mano** (l'auto-detection non regge sul grigio + clutter) → omografia
verso un rettangolo 2:1 → raddrizzamento. **Validazione per auto-consistenza**:
dopo un buon raddrizzamento le palle dovrebbero diventare circolari e uniformi.
**Esito**: su un'immagine **pulita** (tavolo intero, angoli visibili, `out-1-101`) con
angoli marcati bene (griglia di coordinate; attenzione: la buca `MURREY` è la
**centrale**, non un angolo) e aspetto tarato (~2.7:1) il raddrizzamento **funziona**:
palle **circolari** (aspetto_medio ~1.04) e vista dall'alto coerente. Residui onesti:
CV dei diametri ~29% (le palle variano di dimensione secondo la posizione →
**distorsione lente** grandangolare, non correggibile con omografia planare);
aspetto 2.7:1 invece del 2:1 teorico (angoli a mano non perfetti al pixel). Per un
raddrizzamento metrico esatto servirebbe la **calibrazione camera**. Niente ground
truth accoppiata → errore misurato per auto-consistenza (circolarità/uniformità palle).

### QA / utility
- `qa_datasets.py`: conteggi, bilanciamento classi, duplicati (pHash) e leakage.
- `qa_labels.py`: disegna box+numero su immagini campione per verificare le label.
- `analyze_videos.py`: statistiche per video (per decidere lo split).

---

## 6. La geometria della predizione (dettaglio)

- **Vista rettificata**: il tavolo diventa `[0,1000]×[0,500]` (2:1), quindi i
  rimbalzi sono riflessioni asse-allineate (`x`→flip `vx`, `y`→flip `vy`).
- **Collisione (ghost-ball)**: due palle di raggio R si toccano quando i centri
  distano 2R; la palla colpita parte lungo la **retta contatto→centro** (congiungente
  dei centri).
- **Imbucata**: la palla è imbucata se il punto di arrivo su una sponda cade entro
  `CAPTURE` (≈ 2.2·R) da una buca — è la versione "cerchio" del **varco nella sponda**.
- **Catena**: ogni collisione trasferisce il moto alla palla colpita; le palle già
  mosse vengono escluse; si limita la **profondità** (non l'energia, che non c'è).

---

## 7. Materiale video

- `pool 1.mp4`: dall'alto, panno **ciano**, palle vere numerate. Dominio un po'
  diverso dal dataset → qui l'identità **a colore** è preziosa.
- `video2.mp4`: match reale, panno blu, **giocatore in mira** (occlusione) — buche
  robuste + stecca Hough funzionano lo stesso.
- `video3.mp4`, `video4.mp4`, `video5.mp4`: altri tiri dall'alto (blu/ciano).
- Usare **frame fermi** (motion≈0) per la detection; per la stecca serve un frame
  con la **stecca in mira**.

Esempi di esito (catena):
| video | frame | catena | esito |
|---|---|---|---|
| video2 | 69 | `0 → 4` | IN: palla 4 |
| video3 | 0 | `0 → 3 → 10` | OUT |
| video4 | 0 | `0 → 3` (con banco: `0 → 3 → 7`) | OUT (con banco: IN: palla 7) |

---

## 8. Limiti e assunzioni (dichiarati)

- **Nessuna energia/velocità**: la geometria dà il *dove*, non il *quanto*. Non
  sappiamo quanto corre ciascuna palla dopo l'urto, né se si ferma prima.
- **Sensibilità all'angolo della stecca**: nei tiri quasi frontali un piccolo errore
  d'angolo si amplifica sulla direzione della palla colpita. La precisione della
  stecca è il punto fragile n.1.
- **Identità palle su dominio diverso**: le palle occluse restano `?`; per la
  geometria basta la posizione (sono ostacoli anonimi).
- **Rimbalzo non perfetto**: `R` e la linea di sponda sono stimati → errore di pochi
  pixel, non cambia l'esito qualitativo.
- **Naso sponda vs cappello sponda**: il cappello della sponda ha lo stesso ciano del
  campo, quindi `minAreaRect` sborda di pochi px oltre il naso e nella **mappa 2D** una
  palla a ridosso della sponda appare leggermente staccata. Provato a correggere
  (inset fisso, stima del naso dall'ombra): **scartato** perché, data la sensibilità
  del predittore, anche un inset del 2% sposta le buche quel tanto che basta a
  **ribaltare esiti validati** (video3 OUT → IN falso). Conclusione onesta: mappa
  precisa e predizioni corrette sono **accoppiate**; si tiene `minAreaRect` (esiti
  corretti) e si accetta la piccola imprecisione visiva della mappa.
- **Catena semplificata**: profondità limitata, banchi opzionali, niente deviazione
  della battente dopo l'urto, niente collisioni tra due palle entrambe in moto.

---

## 9. Come si esegue

```bash
# Detection singole (su un frame)
python scripts/detect_table.py
python scripts/detect_pockets.py
python scripts/run_core.py "datasets/videos/video3.mp4" 0     # tavolo+buche+palle

# Stecca
python scripts/detect_cue.py "datasets/videos/video3.mp4" 0

# Predizione completa (catena)
python scripts/predict.py "datasets/videos/video2.mp4" 69       # traiettorie pulite
python scripts/predict.py "datasets/videos/video4.mp4" 0 6      # con banchi (max 6 rimbalzi)

# Dataset + training
python scripts/build_merged.py
python scripts/train.py full
```

Gli output si trovano in `output/` (overlay) e `runs/detect/` (training).
