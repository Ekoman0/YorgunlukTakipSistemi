
import firebase_admin
from firebase_admin import credentials, messaging
import time
import os


# =====================================================
# BURAYA DOLDURUN
# =====================================================

# Firebase Service Account JSON dosyasının yolu
SERVICE_ACCOUNT_PATH = "firebase-service-account.json"

# Telefondaki FCM token (Flutter uygulamasından kopyalayın)
# Uygulama > Ayarlar > "Token'ı Kopyala" butonuna basın
DEVICE_TOKEN = "BURAYA_TELEFON_FCM_TOKEN_YAPISTIRILACAK"

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
            print("[UYARI] Firebase bildirimler devre dışı. FIREBASE_KURULUM.md dosyasını okuyun.")
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
    """FCM ile Android telefona bildirim gönder."""
    global _last_sent

    if DEVICE_TOKEN == "BURAYA_TELEFON_FCM_TOKEN_YAPISTIRILACAK":
        return

    now = time.time()
    if alert_type in _last_sent and (now - _last_sent[alert_type]) < COOLDOWN_SECONDS:
        return

    if not _init_firebase():
        return

    try:
        data = {
            "alert_type": alert_type,
            "message": body,
            "timestamp": str(int(now)),
        }
        if extra_data:
            data.update({k: str(v) for k, v in extra_data.items()})

        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    channel_id="yorgunluk_alerts",
                    priority="max",
                    default_sound=True,
                    default_vibrate_timings=True,
                    visibility="public",
                    notification_count=1,
                ),
            ),
            data=data,
            token=DEVICE_TOKEN,
        )

        response = messaging.send(message)
        _last_sent[alert_type] = now
        print(f"[FCM] Bildirim gönderildi ({alert_type}): {response}")

    except Exception as e:
        print(f"[FCM HATA] Bildirim gönderilemedi: {e}")


def send_tired_alert(closed_duration: float):
    send_alert("tired", "😴 DİKKAT! Göz Yorgunluğu",
               f"Gözleriniz {closed_duration:.0f} saniyedir kapalı! Teneffüs alın.",
               {"is_tired": "true"})


def send_yawn_alert():
    send_alert("yawning", "🥱 UYARI: Esneme Tespit Edildi",
               "Esneme başladı. Uyku bastırmadan önce mola verin!",
               {"is_yawning": "true"})


def send_head_tilt_alert():
    send_alert("head_tilt", "📐 UYARI: Baş Eğik",
               "Baş pozisyonunuz bozuk. Dik oturun!",
               {"is_head_tilted": "true"})


def send_smoking_alert():
    send_alert("smoking", "🚬 YASAK: Sigara/Duman Algılandı",
               "Araç içinde sigara tespit edildi! Hemen durdurun.",
               {"is_smoking": "true"})


def send_phone_alert():
    send_alert("phone", "📱 YASAK: Telefon Kullanımı",
               "Sürüş sırasında telefon kullanımı tespit edildi!",
               {"is_phone": "true"})


def send_face_lost_alert(duration: float):
    send_alert("face_lost", "👤 TEHLİKE: Sürücü Görünmüyor!",
               f"Sürücü {duration:.0f} saniyedir kameradan kayboldu!",
               {"is_face_lost": "true"})
