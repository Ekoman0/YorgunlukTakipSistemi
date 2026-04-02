# Firebase Kurulum Rehberi

Bu rehber, Yorgunluk Takip Sistemi'ni Firebase ile entegre etmek için adım adım talimatları içerir.

---

## 1. Firebase Projesi Oluşturun

1. [Firebase Console](https://console.firebase.google.com)'a gidin
2. **"Proje Ekle"** butonuna tıklayın
3. Proje adı girin: `YorgunlukTakipSistemi`
4. Google Analytics'i isteğe göre etkinleştirin → **Proje Oluştur**

---

## 2. Android Uygulaması Ekleyin

1. Firebase Console'da projenizi açın
2. **"Android"** ikonuna tıklayın
3. Şu bilgileri girin:
   - **Android paket adı**: `com.yorgunluktakip.yorgunluk_takip`
   - **Uygulama adı**: Sürücü Güvenlik (isteğe bağlı)
4. **"Uygulamayı Kaydet"** butonuna tıklayın
5. `google-services.json` dosyasını indirin
6. İndirilen dosyayı şuraya kopyalayın:
   ```
   mobile_app/android/app/google-services.json
   ```

---

## 3. Service Account Key İndirin (Python için)

1. Firebase Console'da **Proje Ayarları** (⚙️ çark) > **Hizmet Hesapları** sekmesine gidin
2. **"Yeni özel anahtar oluştur"** butonuna tıklayın
3. İndirilen JSON dosyasını şuraya kopyalayın:
   ```
   YorgunlukTakipSistemi-main/firebase-service-account.json
   ```

---

## 4. Python Bağımlılığını Kurun

```bash
pip install -r requirements_fcm.txt
```

---

## 5. FCM Token'ı Ayarlayın

1. Flutter uygulamasını Android telefonunuza yükleyin
2. Uygulamayı açın → **Ayarlar** sekmesine gidin
3. "Cihaz Token'ı" bölümündeki token'ı kopyalayın
4. `firebase_notifier.py` dosyasını açın:

```python
# Bu satırı bulun ve token'ı yapıştırın:
DEVICE_TOKEN = "BURAYA_TELEFON_FCM_TOKEN_YAPISTIRILACAK"
```

---

## 6. Uygulamayı APK Olarak Derleyin

Android Studio'da `mobile_app` klasörünü açın:

```bash
cd mobile_app
flutter build apk --release
```

APK dosyası şuraya oluşturulur:
```
mobile_app/build/app/outputs/flutter-apk/app-release.apk
```

Bu dosyayı telefonunuza USB veya Bluetooth ile gönderin ve yükleyin.

---

## 7. Test Edin

1. Python backend'i çalıştırın:
   ```bash
   python main.py
   ```
2. Kamera önüne oturun
3. Gözlerinizi ~1 saniye kapatın → **Telefona bildirim gelecek!**

---

## Önemli Notlar

- Telefon ve PC aynı WiFi'da olmak **zorunda değil** (FCM internet üzerinden çalışır)
- Telefon kilitliyken de bildirim gelir (`fullScreenIntent` etkin)
- Her uyarı türü 30 saniyede bir tekrar gönderilir (spam koruması)
- `firebase-service-account.json` dosyasını **kimseyle paylaşmayın** (gizli anahtar!)
