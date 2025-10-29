from ultralytics import YOLO

# Önceden hazırlanmış yolov8n.pt modeli (nano, hızlı)
model = YOLO("yolov5n.pt")

# Dataset yolunu buraya yaz
data_yaml = "data.yaml"

# Eğitim başlat
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,        # GPU varsa batch artırılabilir
    project="smoke_detection", # Eğitim sonucu klasör
    name="run1",      # Run adı
)
