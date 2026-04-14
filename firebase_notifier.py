
import firebase_admin
from firebase_admin import credentials, messaging
import time
import os


# =====================================================
# BURAYA DOLDURUN
# =====================================================

# Firebase Service Account JSON dosyasının yolu
SERVICE_ACCOUNT_PATH = "firebase-service-account.json"

# Telefondaki FCM token (uygulamadan kopyalayın)
DEVICE_TOKEN = "f_fOJFzaTcW_CVoDnPeQeP:APA91bGw4Y7-kfIOh8kP_TWJvX_1HF27Xuhr_cmsklbCbXinLQOetkn3lEG25PjyertlB41x1aMKpoF9vyFHqbGvzIS6yE-GlCUJ3IYODKej3L5neMW1BO8"

# =====================================================

# Son gönderilen uyarı zamanları (spam önlemi)
_last_sent: dict = {}
COOLDOWN_SECONDS = 30  # Aynı uyarıyı 30 sn'de bir gönder

_firebase_initialized = False


def _init_firebase():
    """Firebase'i başlat (bir kez)"""
    global _firebase_initialized
    if not _firebase_initialized:
        if not os.path.exists(SERVICE_ACCOUNT_PATH):
            print(f"[UYARI] Firebase service account dosyası bulunamadı: {SERVICE_ACCOUNT_PATH}")
            print("[UYARI] Firebase bildirimler devre dışı. Kurulum için FIREBASE_SETUP.md okuyun.")
            return False
        
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
            print("[INFO] Firebase başarıyla başlatıldı.")
            return True
        except Exception as e:
            print(f"[HATA] Firebase başlatılamadı: {e}")
            return False
    return True


def send_alert(alert_type: str, title: str, body: str, extra_data: dict = None):
    """
    FCM ile Android telefona bildirim gönder.
    
    Args:
        alert_type: Uyarı tipi ('tired', 'yawning', 'head_tilt', 'smoking', 'phone', 'face_lost')
        title: Bildirim başlığı
        body: Bildirim metni
        extra_data: Ek veri (durum bayrakları vs.)
    """
    global _last_sent
    
    # Firebase token ayarlanmamışsa çık
    if DEVICE_TOKEN == "BURAYA_TELEFON_FCM_TOKEN_YAPISTIRILACAK":
        return
    
    # Cooldown kontrolü
    now = time.time()
    if alert_type in _last_sent and (now - _last_sent[alert_type]) < COOLDOWN_SECONDS:
        return
    
    if not _init_firebase():
        return
    
    try:
        # Veri hazırla
        data = {
            "alert_type": alert_type,
            "message": body,
            "timestamp": str(int(now)),
        }
        if extra_data:
            data.update({k: str(v) for k, v in extra_data.items()})
        
        # FCM mesajı oluştur
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            android=messaging.AndroidConfig(
                priority="high",  # Kilitli ekranda görünsün
                notification=messaging.AndroidNotification(
                    channel_id="yorgunluk_alerts",
                    priority="max",
                    default_sound=True,
                    default_vibrate_timings=True,
                    visibility="public",  # Kilitli ekranda göster
                    notification_count=1,
                ),
            ),
            data=data,
            token=DEVICE_TOKEN,
        )
        
        # Gönder
        response = messaging.send(message)
        _last_sent[alert_type] = now
        print(f"[FCM] Bildirim gönderildi ({alert_type}): {response}")
        
    except Exception as e:
        print(f"[FCM HATA] Bildirim gönderilemedi: {e}")


def send_tired_alert(closed_duration: float):
    send_alert(
        alert_type="tired",
        title="😴 DİKKAT! Göz Yorgunluğu",
        body=f"Gözleriniz {closed_duration:.0f} saniyedir kapalı! Teneffüs alın.",
        extra_data={"is_tired": "true"},
    )


def send_yawn_alert():
    send_alert(
        alert_type="yawning",
        title="🥱 UYARI: Esneme Tespit Edildi",
        body="Esneme başladı. Uyku bastırmadan önce mola verin!",
        extra_data={"is_yawning": "true"},
    )


def send_head_tilt_alert():
    send_alert(
        alert_type="head_tilt",
        title="📐 UYARI: Baş Eğik",
        body="Baş pozisyonunuz bozuk. Dik oturun!",
        extra_data={"is_head_tilted": "true"},
    )


def send_smoking_alert():
    send_alert(
        alert_type="smoking",
        title="🚬 YASAK: Sigara/Duman Algılandı",
        body="Araç içinde sigara tespit edildi! Hemen durdurun.",
        extra_data={"is_smoking": "true"},
    )


def send_phone_alert():
    send_alert(
        alert_type="phone",
        title="📱 YASAK: Telefon Kullanımı",
        body="Sürüş sırasında telefon kullanımı tespit edildi!",
        extra_data={"is_phone": "true"},
    )


def send_face_lost_alert(duration: float):
    send_alert(
        alert_type="face_lost",
        title="👤 TEHLİKE: Sürücü Görünmüyor!",
        body=f"Sürücü {duration:.0f} saniyedir kameradan kayboldu!",
        extra_data={"is_face_lost": "true"},
    )
