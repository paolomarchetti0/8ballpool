"""
Punto 3 - Ispezione del video e estrazione di alcuni frame.

Non fa detection: apre il video, stampa le proprieta' (risoluzione, fps,
numero di frame, durata) e salva qualche frame campione su disco cosi'
possiamo guardarli e scegliere quello del "prima del colpo".
"""
import os
import cv2

VIDEO = "datasets/videos/pool 1.mp4"
OUT_DIR = "output"

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise SystemExit(f"Non riesco ad aprire il video: {VIDEO}")

    # Proprieta' del video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n_frames / fps if fps else 0

    print(f"File      : {VIDEO}")
    print(f"Risoluzione: {w} x {h}")
    print(f"FPS       : {fps:.2f}")
    print(f"Frame tot : {n_frames}")
    print(f"Durata    : {dur:.1f} s")

    # Salviamo alcuni frame campione spalmati sul video, per capirne il contenuto.
    n_samples = 6
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\nFrame campione salvati:")
    for i in range(n_samples):
        # posizioni: evitiamo il primo e l'ultimo frame esatti
        pos = int((i + 1) * n_frames / (n_samples + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {pos}: lettura fallita")
            continue
        path = os.path.join(OUT_DIR, f"sample_{i:02d}_frame{pos:05d}.png")
        cv2.imwrite(path, frame)
        print(f"  {path}  (frame {pos}, t={pos/fps:.1f}s)")

    cap.release()

if __name__ == "__main__":
    main()
