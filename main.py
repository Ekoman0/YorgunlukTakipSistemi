import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
import time
import numpy as np
import os
import sys
import threading
import mediapipe as mp
from ultralytics import YOLO

# --- Kamera Seçim Modülü ---
from camera_selector import select_camera

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
SMOKE_CONSEC_FRAMES = 5   # Az düşürüldü: daha hızlı uyarı
PHONE_CONSEC_FRAMES = 5   # Az düşürüldü: daha hızlı uyarı

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

# --- MediaPipe Hands Yükleme ---
print("[INFO] MediaPipe El Takibi (Hands) yükleniyor (Python 3.10)...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# --- YOLOv8 Telefon Algılama Yükleme ---
print("[INFO] Telefon algılama modeli (YOLOv8s) yükleniyor...")
phone_model = YOLO("yolov8s.pt")

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

# (YOLO önişleme ve ROI fonksiyonları kaldırıldı çünkü artık MediaPipe kullanıyoruz)

# --- Kamera Seçimi (GUI) ---
print("[INFO] Kamera seçim ekranı açılıyor...")
camera_choice = select_camera()

if camera_choice is None:
    print("[INFO] Kamera seçilmedi. Program kapatılıyor.")
    sys.exit(0)

camera_type, camera_source = camera_choice

# Kamera etiket ve açıklamaları
_CAMERA_LABELS = {
    "laptop": ("Laptop Kamerasi", (180, 255, 180)),
    "wifi":   ("iPhone - Wi-Fi",  (180, 220, 255)),
    "usb":    ("iPhone - USB",    (255, 220, 180)),
}
CAMERA_LABEL_TEXT, CAMERA_LABEL_COLOR = _CAMERA_LABELS.get(
    camera_type, ("Kamera", (255, 255, 255))
)

# --- Kamera Akışı ve Çözünürlük Ayarı ---
print(f"[INFO] Kamera başlatılıyor: {camera_type} → {camera_source}")

# ─────────────────────────────────────────────────────────────────────────────
# Threaded VideoCapture: kamera okuma ayrı thread'de döner,
# ana döngü frame bekliyerek bloklanmaz → FPS önemli ölçüde artar.
# ─────────────────────────────────────────────────────────────────────────────
class ThreadedCapture:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.cap.release()

vs = ThreadedCapture(camera_source)
time.sleep(0.5)  # thread'in ilk frame'i okuması için kısa bekleme

if not vs.isOpened():
    print("[HATA] Kamera açılamadı! Lütfen bağlantıyı kontrol edin.")
    import tkinter as tk
    from tkinter import messagebox
    _r = tk.Tk(); _r.withdraw()
    messagebox.showerror(
        "Kamera Hatası",
        f"Kamera açılamadı!\n\nKaynak: {camera_source}\n\n"
        "• Wi-Fi yöntemi için: IP/Port doğru mu? Uygulama açık mı?\n"
        "• USB yöntemi için: Camo/EpocCam kurulu ve iPhone bağlı mı?\n"
        "• Laptop kamerası için: Kamera başka bir uygulama tarafından kullanılıyor mu?"
    )
    _r.destroy()
    sys.exit(1)

# ───────────────────────────────────────────────────────────────────────────
# FPS ve YOLO ayarları
# ───────────────────────────────────────────────────────────────────────────
YOLO_SKIP_FRAMES = 5
_yolo_frame_counter = 0
_fps_frame_counter  = 0
_fps_start = time.time()
_fps_value = 0.0

# Telefon Bounding Box Cache
_last_phone_boxes = []

cv2.namedWindow("Yorgunluk ve Duman & Telefon Algilama", cv2.WINDOW_NORMAL)

while True:
    ret, frame = vs.read()
    if not ret or frame is None:
        time.sleep(0.005)
        continue

    # FPS hesapla
    _fps_frame_counter += 1
    elapsed = time.time() - _fps_start
    if elapsed >= 0.5:
        _fps_value = _fps_frame_counter / elapsed
        _fps_frame_counter = 0
        _fps_start = time.time()

    (h, w) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = detector(gray, 0)
    
    # --- MediaPipe ile El ve Parmak Takibi ---
    hand_results = hands.process(rgb_frame)
    hand_hulls = [] # Algılanan ellerin sınırları (collider)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 21 noktanın hepsini piksel koordinatına çevir
            points = []
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append([px, py])
                
            points = np.array(points, dtype=np.int32)
            
            # Tüm parmakları sarmalayan en dış şekli (Convex Hull) oluştur
            hull = cv2.convexHull(points)
            hand_hulls.append(hull)

    # --- YOLO ile Telefon Tespiti (Mesajlaşma/Video için) ---
    _yolo_frame_counter += 1
    if _yolo_frame_counter % YOLO_SKIP_FRAMES == 0:
        # Sadece 67 (cell phone) sınıfını arıyoruz. 
        # imgsz=320 ile performansı koruyoruz.
        phone_res = phone_model(frame, conf=0.45, imgsz=320, classes=[67], verbose=False)[0]
        _last_phone_boxes = []
        for box in phone_res.boxes.xyxy:
            _last_phone_boxes.append(box.cpu().numpy().astype(int))

    is_smoking = False
    is_tired = False
    is_yawning = False
    is_phone = False
    is_texting = False # Yeni mesajlaşma durumu
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
        LAST_FACE_TIME = time.time()
        
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            # --- Yüz Genişliği ve Referans Noktaları (Dlib) ---
            # Yüz genişliğini hesaplamak için şakak hizasındaki en geniş noktaları (0 ve 16) kullanıyoruz
            face_width = distance.euclidean(shape[0], shape[16])
            
            # Kulak noktalarını tam kulağın ortasına (1 ve 15) alıyoruz.
            left_ear_pt = shape[1]
            right_ear_pt = shape[15]
            mouth_center = mouth.mean(axis=0)

            # Hedef noktaları ekranda mavi göster (Hata ayıklama / görsellik)
            cv2.circle(frame, (int(mouth_center[0]), int(mouth_center[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(left_ear_pt[0]), int(left_ear_pt[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(right_ear_pt[0]), int(right_ear_pt[1])), 5, (255, 0, 0), -1)

            # --- Gerçek El Collider Kontrolü (MediaPipe) ---
            # Tolerans: Collider'ın kaç piksel dışına kadar algılasın (yüz genişliğinin %10'u kadar)
            tolerance = face_width * 0.10
            
            for hull in hand_hulls:
                # Ağız noktasının el collider'ına (hull) olan mesafesini hesapla
                # Eğer değer pozitifse: Nokta içeride
                # Eğer değer negatifse: Nokta dışarıda (uzaklık olarak döner)
                mouth_dist = cv2.pointPolygonTest(hull, (float(mouth_center[0]), float(mouth_center[1])), True)
                l_ear_dist = cv2.pointPolygonTest(hull, (float(left_ear_pt[0]), float(left_ear_pt[1])), True)
                r_ear_dist = cv2.pointPolygonTest(hull, (float(right_ear_pt[0]), float(right_ear_pt[1])), True)
                
                collider_color = (0, 255, 0) # Normalde yeşil collider
                
                # Ağız elin içindeyse veya çok yakınındaysa (tolerans)
                if mouth_dist >= -tolerance:
                    is_smoking = True
                    collider_color = (0, 0, 255) # Kırmızı
                    
                # Kulak elin içindeyse veya çok yakınındaysa
                elif l_ear_dist >= -tolerance or r_ear_dist >= -tolerance:
                    is_phone = True
                    collider_color = (0, 165, 255) # Turuncu
                    
                # Eli tam şekliyle çiz (Kusursuz parmak kenarları)
                cv2.drawContours(frame, [hull], -1, collider_color, 2)
                # İçini hafif şeffaf doldurmak istersen: cv2.fillPoly(frame, [hull], (0,255,0)) yapabilirsin ama çizgili daha iyi.
                
                # Etiket
                # Hull'un en üst noktasını bulup yazıyı oraya yazalım
                top_point = tuple(hull[hull[:, :, 1].argmin()][0])
                cv2.putText(frame, "EL COLLIDER", (top_point[0] - 40, top_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, collider_color, 2)

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

            # --- Telefona Bakma (Texting) Kontrolü ---
            # Eğer kamera açısında bir telefon varsa VE sürücünün başı öne eğikse
            # (veya telefon algılanmışsa ve el telefonun üzerindeyse, 
            # şimdilik sadece telefon + baş eğikliği yeterli)
            if len(_last_phone_boxes) > 0 and is_head_tilted:
                is_texting = True

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


    # ─────────────────────────────────────────────────────────────────
    # YOLO – Her YOLO_SKIP_FRAMES frame'de bir çalışır.
    # imgsz=320  → varsayılan 640'dan ~4x daha hızlı inference
    # verbose=False → console spam yok, işlemci biraz rahatlar
    # ─────────────────────────────────────────────────────────────────
    # --- (Eski YOLO kodları silindi) ---
    
    # Telefon / Sigara uyarılarını ekrana yazdır
    if is_smoking:
        cv2.putText(frame, "EL-AGIZ (SIGARA?)", (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        current_y += line_spacing
        
    if is_phone:
        cv2.putText(frame, "EL-KULAK (TELEFONLA KONUSMA?)", (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        current_y += line_spacing

    if is_texting:
        cv2.putText(frame, "TELEFONA BAKIYOR (MESAJ/VIDEO)!", (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        current_y += line_spacing

    # YOLO ile bulunan telefonu ekranda göster
    for (x1, y1, x2, y2) in _last_phone_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, "TELEFON", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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

    # --- FPS + Kamera tipi etiketi (sağ alt köşe) ---
    overlay_text = f"{CAMERA_LABEL_TEXT}  |  {_fps_value:.1f} FPS"
    label_size, _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lw, lh = label_size
    lx = w - lw - 12
    ly = h - 10
    cv2.rectangle(frame, (lx - 6, ly - lh - 6), (lx + lw + 6, ly + 4), (20, 20, 40), -1)
    cv2.putText(frame, overlay_text, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CAMERA_LABEL_COLOR, 1, cv2.LINE_AA)

    cv2.imshow("Yorgunluk ve Duman & Telefon Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()