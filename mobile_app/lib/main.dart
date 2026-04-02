import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'screens/dashboard_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/alert_history_screen.dart';
import 'services/notification_service.dart';
import 'theme/app_theme.dart';

// Background message handler (top-level function zorunlu)
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
  await NotificationService.instance.showNotification(
    title: message.notification?.title ?? 'Uyarı!',
    body: message.notification?.body ?? '',
    alertType: message.data['alert_type'] ?? 'general',
  );
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Firebase başlat
  await Firebase.initializeApp();
  
  // Background handler'ı kaydet
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  
  // Notification service başlat
  await NotificationService.instance.initialize();
  
  runApp(const YorgunlukTakipApp());
}

class YorgunlukTakipApp extends StatelessWidget {
  const YorgunlukTakipApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sürücü Güvenlik Sistemi',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      initialRoute: '/',
      routes: {
        '/': (context) => const MainNavigationScreen(),
        '/settings': (context) => const SettingsScreen(),
        '/alerts': (context) => const AlertHistoryScreen(),
      },
    );
  }
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    const DashboardScreen(),
    const AlertHistoryScreen(),
    const SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    _setupFCMListeners();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    await FirebaseMessaging.instance.requestPermission(
      alert: true,
      badge: true,
      sound: true,
      criticalAlert: true,
    );
  }

  void _setupFCMListeners() {
    // Uygulama açıkken gelen bildirimler
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      NotificationService.instance.showNotification(
        title: message.notification?.title ?? 'Uyarı!',
        body: message.notification?.body ?? '',
        alertType: message.data['alert_type'] ?? 'general',
      );
      
      // Dashboard'u güncelle (alert bilgisini yayınla)
      AlertEventBus.instance.emit(message.data);
    });

    // Bildirime tıklanarak uygulama açıldıysa
    FirebaseMessaging.onMessageOpenedApp.listen((RemoteMessage message) {
      Navigator.pushReplacementNamed(context, '/alerts');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          border: Border(
            top: BorderSide(
              color: AppTheme.accentBlue.withOpacity(0.3),
              width: 1,
            ),
          ),
        ),
        child: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) => setState(() => _currentIndex = index),
          backgroundColor: AppTheme.darkSurface,
          selectedItemColor: AppTheme.accentBlue,
          unselectedItemColor: AppTheme.textSecondary,
          type: BottomNavigationBarType.fixed,
          selectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 11,
          ),
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.dashboard_rounded),
              label: 'Panel',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.history_rounded),
              label: 'Geçmiş',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings_rounded),
              label: 'Ayarlar',
            ),
          ],
        ),
      ),
    );
  }
}

// Basit event bus - FCM mesajlarını dinleyen widget'lara iletir
class AlertEventBus {
  static final AlertEventBus instance = AlertEventBus._internal();
  AlertEventBus._internal();

  final List<Function(Map<String, dynamic>)> _listeners = [];

  void addListener(Function(Map<String, dynamic>) listener) {
    _listeners.add(listener);
  }

  void removeListener(Function(Map<String, dynamic>) listener) {
    _listeners.remove(listener);
  }

  void emit(Map<String, dynamic> data) {
    for (final listener in _listeners) {
      listener(data);
    }
  }
}
