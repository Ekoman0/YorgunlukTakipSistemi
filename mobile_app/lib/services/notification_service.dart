import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class NotificationService {
  static final NotificationService instance = NotificationService._internal();
  NotificationService._internal();

  final FlutterLocalNotificationsPlugin _plugin =
      FlutterLocalNotificationsPlugin();

  Future<void> initialize() async {
    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const InitializationSettings initSettings = InitializationSettings(
      android: androidSettings,
    );

    await _plugin.initialize(initSettings);

    // Android notification channel (yüksek öncelikli - kilitli ekran için)
    const AndroidNotificationChannel channel = AndroidNotificationChannel(
      'yorgunluk_alerts',
      'Sürücü Güvenlik Uyarıları',
      description: 'Yorgunluk ve güvenlik ihlali uyarıları',
      importance: Importance.max,
      playSound: true,
      enableVibration: true,
      showBadge: true,
    );

    await _plugin
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(channel);
  }

  Future<void> showNotification({
    required String title,
    required String body,
    required String alertType,
  }) async {
    final Color ledColor = _getAlertColor(alertType);

    final AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'yorgunluk_alerts',
      'Sürücü Güvenlik Uyarıları',
      channelDescription: 'Yorgunluk ve güvenlik ihlali uyarıları',
      importance: Importance.max,
      priority: Priority.max,
      fullScreenIntent: true, // Kilitli ekranda tam ekran aç
      ticker: title,
      styleInformation: BigTextStyleInformation(body),
      color: ledColor,
      enableLights: true,
      ledColor: ledColor,
      ledOnMs: 500,
      ledOffMs: 500,
    );

    final NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );

    await _plugin.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      details,
    );
  }

  Color _getAlertColor(String alertType) {
    switch (alertType) {
      case 'tired':
      case 'face_lost':
      case 'smoking':
        return const Color(0xFFEF4444);
      case 'phone':
        return const Color(0xFF6366F1);
      case 'yawning':
      case 'head_tilt':
        return const Color(0xFFF59E0B);
      default:
        return const Color(0xFF3B82F6);
    }
  }
}
