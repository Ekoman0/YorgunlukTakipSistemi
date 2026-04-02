import cv2
import dlib
import torch
from ultralytics import YOLO
import imutils
from scipy.spatial import distance
from imutils import face_utils
import time
import numpy as np
import os

# --- Firebase FCM Bildirimi ---
try:
    import firebase_notifier as fcm
    FCM_ENABLED = True
    print("[INFO] Firebase FCM modülü yüklendi.")
except ImportError:
    FCM_ENABLED = False
    print("[UYARI] firebase_notifier bulunamadı. FCM bildirimleri devre dışı.")

# --- Yorgunluk ve Duman Algılama Ayarları ---
EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME_LIMIT = 1.0
MOUTH_MAR_THRESHOLD = 0.70
YAWN_CONSEC_FRAMES = 15
SMOKE_CONSEC_FRAMES = 10
PHONE_CONSEC_FRAMES = 10

# --- Baş Eğilmesi Ayarları ---
HEAD_TILT_THRESHOLD = 20  # Derece cinsinden eşik değeri
HEAD_TILT_CONSEC_FRAMES = 15  # Kaç frame boyunca eğik kalmalı

# --- YENİ EKLENEN AYAR ---
FACE_LOST_TIME_LIMIT = 10.0 # Yüz 10 saniyeden fazla algılanmazsa uyarı ver

# --- Global Değişkenler ve Sayaçlar ---
EYE_CLOSED_START_TIME = None
YAWN_COUNTER = 0
SMOKE_COUNTER = 0
PHONE_COUNTER = 0
HEAD_TILT_COUNTER = 0
# YENİ EKLENEN DEĞİŞKEN
LAST_FACE_TIME = time.time() # Program başladığında yüzün algılandığı ilk an

# --- Dlib Modellerini Yükleme ---
print("[INFO] Yüz algılayıcı ve işaret noktası tahmincisi yükleniyor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- YOLO Modellerini Yükle ---
print("[INFO] Duman algılama modeli yükleniyor...")
smoke_model = YOLO("best.pt")
print("Smoke model sınıfları:", smoke_model.names)

print("[INFO] Sigara algılama modeli yükleniyor...")
# YENİ EKLENEN MODEL
cigarette_model = YOLO("cigarettedetect.pt")
print("Sigara modeli sınıfları:", cigarette_model.names)

print("[INFO] Telefon algılama modeli (YOLOv8n) yükleniyor...")
phone_model = YOLO("yolov8n.pt")
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

def calculate_head_tilt(shape):
    """ Başın yatay eğimini (Z ekseni etrafında dönme) hesaplar. """
    left_eye = shape[36]
    right_eye = shape[45]
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle

def calculate_head_pitch(shape):
    """ Baş öne/arkaya eğilme açısını (pitch - X ekseni etrafında dönme) hesaplar. """
    nose_tip = shape[30]
    left_eye_center = shape[36:42].mean(axis=0)
    right_eye_center = shape[42:48].mean(axis=0)
    eye_center = ((left_eye_center + right_eye_center) / 2).astype(int)
    chin = shape[8]
    face_height = distance.euclidean(eye_center, chin)
    nose_length = distance.euclidean(eye_center, nose_tip)
    
    if face_height > 0:
        pitch_ratio = nose_length / face_height
        pitch_angle = (pitch_ratio - 0.35) * 100
        return pitch_angle
    return 0

# --- Kamera Akışı ve Çözünürlük Ayarı ---
print("[INFO] Kamera başlatılıyor...")
vs = cv2.VideoCapture(0)

# Çözünürlüğü 640x480 olarak ayarla (Daha iyi performans için)
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Yorgunluk ve Duman & Telefon Algilama", cv2.WINDOW_NORMAL)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    (h, w) = frame.shape[:2] # Çerçevenin genişlik ve yüksekliğini al
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    is_smoking = False
    is_tired = False
    is_yawning = False
    is_phone = False
    is_head_tilted = False
    is_face_lost = False # Yüz kaybı bayrağı
    closed_duration = 0.0
    
    # Sağ üst köşeden başlama pozisyonu
    display_x = w - 250
    display_y = 30
    line_spacing = 30
    current_y = display_y

    # --- Yüz Algılama ve Yorgunluk Tespiti ---
    if len(faces) > 0:
        # Yüz algılandı: Son algılanma zamanını güncelle
        LAST_FACE_TIME = time.time()
        
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mouthMAR = mouth_aspect_ratio(mouth)

            # Baş eğilme ve pitch açısı
            tilt_angle = calculate_head_tilt(shape)
            pitch_angle = calculate_head_pitch(shape)
            
            abs_tilt = abs(tilt_angle)
            abs_pitch = abs(pitch_angle)
            
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

            # Göz yorgunluğu kontrolü
            if ear < EAR_THRESHOLD:
                if EYE_CLOSED_START_TIME is None:
                    EYE_CLOSED_START_TIME = time.time()
                closed_duration = time.time() - EYE_CLOSED_START_TIME
                if closed_duration >= CLOSED_EYE_TIME_LIMIT:
                    is_tired = True
            else:
                EYE_CLOSED_START_TIME = None

            # Esneme kontrolü
            if mouthMAR > MOUTH_MAR_THRESHOLD:
                YAWN_COUNTER += 1
                if YAWN_COUNTER >= YAWN_CONSEC_FRAMES:
                    is_yawning = True
            else:
                YAWN_COUNTER = 0

            # Baş eğilmesi kontrolü
            if abs_tilt > HEAD_TILT_THRESHOLD or abs_pitch > HEAD_TILT_THRESHOLD:
                HEAD_TILT_COUNTER += 1
                if HEAD_TILT_COUNTER >= HEAD_TILT_CONSEC_FRAMES:
                    is_head_tilted = True
            else:
                HEAD_TILT_COUNTER = 0

            # Sağ üst köşeye Bilgi metinleri
            cv2.putText(frame, f"EAR: {ear:.2f}", (display_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            current_y += line_spacing
            cv2.putText(frame, f"MAR: {mouthMAR:.2f}", (display_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            current_y += line_spacing
            cv2.putText(frame, f"Yatay Aci: {tilt_angle:.1f}", (display_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            current_y += line_spacing
            cv2.putText(frame, f"Dikey Aci: {pitch_angle:.1f}", (display_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            current_y += line_spacing
            cv2.putText(frame, f"Kapali Sure: {closed_duration:.2f} sn", (display_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            current_y += line_spacing
    
    # --- Yüz Kaybı Kontrolü ---
    else:
        time_since_last_face = time.time() - LAST_FACE_TIME
        if time_since_last_face >= FACE_LOST_TIME_LIMIT:
            is_face_lost = True
        
        # Yüz kayıp süresini ekranda göster
        cv2.putText(frame, f"Yuz Kayip: {time_since_last_face:.1f} sn", (display_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        current_y += line_spacing


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

    # --- YOLO ile Sigara Algılama (YENİ EKLENDİ) ---
    cigarette_results = cigarette_model(frame, conf=0.30)[0]
    for box, cls in zip(cigarette_results.boxes.xyxy, cigarette_results.boxes.cls):
        is_smoking = True # Sigara algılandığında duman bayrağını da tetikler
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "SIGARA", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- YOLO ile Telefon Algılama ---
    phone_results = phone_model(frame, conf=0.55, classes=[67])[0]
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
    warning_text = ""
    warning_color = (0, 255, 255) 
    
    # Uyarı Hiyerarşisi
    if is_face_lost:
        warning_text = f"TEHLIKE! Yuz Kaybi ({time_since_last_face:.1f} sn)!"
        warning_color = (0, 0, 255) # Kırmızı
    elif is_tired:
        warning_text = "DIKKAT! Goz Yorgunlugu!"
        warning_color = (0, 0, 255) # Kırmızı
    elif is_yawning:
        warning_text = "DIKKAT! Esneme Basladi!"
        warning_color = (0, 165, 255) # Turuncu
    elif is_head_tilted:
        warning_text = "DIKKAT! Bas Egik - Duz Oturun!"
        warning_color = (0, 128, 255) 
    elif is_smoking and SMOKE_COUNTER >= SMOKE_CONSEC_FRAMES:
        warning_text = "YASAK! Sigara/Duman Algilandi!"
        warning_color = (0, 0, 255) # Kırmızı
    elif is_phone and PHONE_COUNTER >= PHONE_CONSEC_FRAMES:
        warning_text = "YASAK! Telefon Algilandi!"
        warning_color = (255, 0, 0) # Mavi
    elif is_smoking:
        warning_text = "Sigara/Duman Algilaniyor"
        warning_color = (0, 255, 255) # Sarı
    elif is_phone:
        warning_text = "Telefon Algilaniyor"
        warning_color = (255, 255, 0) # Açık Mavi

    # Uyarı metnini sol üst köşeye yazdır
    if warning_text:
        cv2.putText(frame, warning_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)

    # --- FCM Bildirimleri Gönder ---
    if FCM_ENABLED:
        if is_face_lost:
            fcm.send_face_lost_alert(time_since_last_face)
        elif is_tired:
            fcm.send_tired_alert(closed_duration)
        elif is_yawning:
            fcm.send_yawn_alert()
        elif is_head_tilted:
            fcm.send_head_tilt_alert()
        elif is_smoking and SMOKE_COUNTER >= SMOKE_CONSEC_FRAMES:
            fcm.send_smoking_alert()
        elif is_phone and PHONE_COUNTER >= PHONE_CONSEC_FRAMES:
            fcm.send_phone_alert()

    cv2.imshow("Yorgunluk ve Duman & Telefon Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()