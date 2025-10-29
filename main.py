import cv2
import dlib
import torch
from ultralytics import YOLO
import imutils
from scipy.spatial import distance
from imutils import face_utils
import time
import numpy as np

# --- Yorgunluk ve Duman Algılama Ayarları ---
EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME_LIMIT = 1.0
MOUTH_MAR_THRESHOLD = 0.70
YAWN_CONSEC_FRAMES = 15
SMOKE_CONSEC_FRAMES = 10
PHONE_CONSEC_FRAMES = 10  # telefon tespiti için ek sayaç

# --- Global Değişkenler ve Sayaçlar ---
EYE_CLOSED_START_TIME = None
YAWN_COUNTER = 0
SMOKE_COUNTER = 0
PHONE_COUNTER = 0

# --- Dlib Modellerini Yükleme ---
print("[INFO] Yüz algılayıcı ve işaret noktası tahmincisi yükleniyor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- YOLO Modellerini Yükle ---
print("[INFO] Duman algılama modeli yükleniyor...")
smoke_model = YOLO("best.pt")  # Smoke modeli
print("Smoke model sınıfları:", smoke_model.names)

print("[INFO] Telefon algılama modeli (YOLOv8n) yükleniyor...")
phone_model = YOLO("yolov8n.pt")  # Telefon modeli (COCO dataset)
print("Telefon model sınıfları:", phone_model.names)

# --- Kilit Noktaları Tanımlama ---
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# --- Fonksiyonlar ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# --- Kamera Akışı ---
print("[INFO] Kamera başlatılıyor...")
vs = cv2.VideoCapture(0)
cv2.namedWindow("Yorgunluk ve Duman & Telefon Algilama", cv2.WINDOW_NORMAL)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    is_smoking = False
    is_tired = False
    is_yawning = False
    is_phone = False

    # --- Yorgunluk tespiti ---
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mouthMAR = mouth_aspect_ratio(mouth)

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            if EYE_CLOSED_START_TIME is None:
                EYE_CLOSED_START_TIME = time.time()
            closed_duration = time.time() - EYE_CLOSED_START_TIME
            if closed_duration >= CLOSED_EYE_TIME_LIMIT:
                is_tired = True
            cv2.putText(frame, f"Kapali: {closed_duration:.2f} sn", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            EYE_CLOSED_START_TIME = None

        if mouthMAR > MOUTH_MAR_THRESHOLD:
            YAWN_COUNTER += 1
            if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                is_yawning = True
        else:
            YAWN_COUNTER = 0

        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mouthMAR:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # --- YOLO ile Duman Algılama ---
    smoke_results = smoke_model(frame, conf=0.55)[0]

    for box, cls in zip(smoke_results.boxes.xyxy, smoke_results.boxes.cls):
        label = smoke_model.names[int(cls)]
        if "0" in label.lower():
            is_smoking = True
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "SMOKE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- YOLO ile Telefon Algılama ---
    phone_results = phone_model(frame, conf=0.55, classes=[67])[0]  # 67 = 'cell phone' COCO sınıf ID'si
    for box, cls in zip(phone_results.boxes.xyxy, phone_results.boxes.cls):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        is_phone = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, "CELL PHONE", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # Sayaçlar
    if is_smoking:
        SMOKE_COUNTER += 1
    else:
        SMOKE_COUNTER = 0

    if is_phone:
        PHONE_COUNTER += 1
    else:
        PHONE_COUNTER = 0

    # --- Uyarılar ---
    if is_tired or is_yawning or (is_smoking and SMOKE_COUNTER >= SMOKE_CONSEC_FRAMES) or (is_phone and PHONE_COUNTER >= PHONE_CONSEC_FRAMES):
        cv2.putText(frame, "DIKKAT! ACIL UYARI!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif is_tired:
        cv2.putText(frame, "Gozyorgunlugu basladi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif is_yawning:
        cv2.putText(frame, "Esneme basladi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif is_smoking:
        cv2.putText(frame, "Duman Algilandi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif is_phone:
        cv2.putText(frame, "Telefon Algilandi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    cv2.imshow("Yorgunluk ve Duman & Telefon Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()
