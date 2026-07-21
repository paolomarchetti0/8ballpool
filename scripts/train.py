"""
Training YOLO sulle palle (dataset unito, split per video).

Transfer learning dai pesi generici YOLOv8. Parametri in cima per passare
facilmente da PROVA (pochi epoch, modello nano) a COMPLETO.

Uso:
  python scripts/train.py            # prova (default)
  python scripts/train.py full       # training completo
"""
import sys
from ultralytics import YOLO

DATA = "datasets/merged/data.yaml"

TRIAL = dict(model="yolov8n.pt", epochs=3,  imgsz=640, batch=16, name="trial")
FULL  = dict(model="yolov8s.pt", epochs=80, imgsz=640, batch=16, name="full")


def main():
    cfg = FULL if (len(sys.argv) > 1 and sys.argv[1] == "full") else TRIAL
    print(f"Config: {cfg}")
    model = YOLO(cfg["model"])
    model.train(
        data=DATA,
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=0,               # GPU (RTX 3060 Ti)
        name=cfg["name"],
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
