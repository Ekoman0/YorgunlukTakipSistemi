import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:vibration/vibration.dart';
import '../main.dart';
import '../theme/app_theme.dart';
import '../widgets/status_card.dart';
import '../services/notification_service.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with TickerProviderStateMixin {
  // Durum değerleri
  bool _isTired = false;
  bool _isYawning = false;
  bool _isHeadTilted = false;
  bool _isSmoking = false;
  bool _isPhone = false;
  bool _isFaceLost = false;
  bool _isConnected = false;

  String _currentAlert = '';
  String _alertType = '';
  String? _fcmToken;
  DateTime? _lastAlertTime;
  
  late AnimationController _pulseController;
  late AnimationController _glowController;
  late Animation<double> _pulseAnim;
  late Animation<double> _glowAnim;
  
  final AudioPlayer _audioPlayer = AudioPlayer();
  final List<Map<String, dynamic>> _alertHistory = [];

  @override
  void initState() {
    super.initState();
    
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    )..repeat(reverse: true);
    
    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);
    
    _pulseAnim = Tween<double>(begin: 0.95, end: 1.05).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
    
    _glowAnim = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _glowController, curve: Curves.easeInOut),
    );

    _loadFCMToken();
    _setupAlertListener();
  }

  Future<void> _loadFCMToken() async {
    final token = await FirebaseMessaging.instance.getToken();
    setState(() => _fcmToken = token);
    
    // Token'ı SharedPreferences'a kaydet
    final prefs = await SharedPreferences.getInstance();
    if (token != null) {
      await prefs.setString('fcm_token', token);
      setState(() => _isConnected = true);
    }
  }

  void _setupAlertListener() {
    AlertEventBus.instance.addListener(_onAlertReceived);
  }

  void _onAlertReceived(Map<String, dynamic> data) {
    final alertType = data['alert_type'] ?? 'general';
    final message = data['message'] ?? '';
    
    setState(() {
      _currentAlert = message;
      _alertType = alertType;
      _lastAlertTime = DateTime.now();
      
      // Durum güncelle
      _isTired = data['is_tired'] == 'true';
      _isYawning = data['is_yawning'] == 'true';
      _isHeadTilted = data['is_head_tilted'] == 'true';
      _isSmoking = data['is_smoking'] == 'true';
      _isPhone = data['is_phone'] == 'true';
      _isFaceLost = data['is_face_lost'] == 'true';
      
      // Geçmişe ekle
      _alertHistory.insert(0, {
        'time': DateTime.now(),
        'message': message,
        'type': alertType,
      });
      if (_alertHistory.length > 50) _alertHistory.removeLast();
    });
    
    _triggerAlertEffects(alertType);
    
    // 5 saniye sonra uyarıyı temizle
    Future.delayed(const Duration(seconds: 5), () {
      if (mounted) {
        setState(() {
          _currentAlert = '';
          _alertType = '';
        });
      }
    });
  }

  Future<void> _triggerAlertEffects(String alertType) async {
    final isCritical = ['tired', 'face_lost', 'smoking', 'phone'].contains(alertType);
    
    // Titreşim
    if (await Vibration.hasVibrator() ?? false) {
      if (isCritical) {
        Vibration.vibrate(pattern: [0, 500, 100, 500, 100, 500]);
      } else {
        Vibration.vibrate(duration: 300);
      }
    }
  }

  Color _getAlertColor() {
    switch (_alertType) {
      case 'tired':
      case 'face_lost':
        return AppTheme.dangerRed;
      case 'smoking':
        return AppTheme.criticalRed;
      case 'phone':
        return AppTheme.phoneBlue;
      case 'yawning':
      case 'head_tilt':
        return AppTheme.warningOrange;
      default:
        return AppTheme.accentBlue;
    }
  }

  @override
  void dispose() {
    AlertEventBus.instance.removeListener(_onAlertReceived);
    _pulseController.dispose();
    _glowController.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.darkBackground,
      body: CustomScrollView(
        slivers: [
          _buildAppBar(),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Aktif Uyarı Alanı
                  if (_currentAlert.isNotEmpty) ...[
                    _buildActiveAlert(),
                    const SizedBox(height: 16),
                  ],
                  
                  // FCM Token / Bağlantı Durumu
                  _buildConnectionCard(),
                  const SizedBox(height: 20),
                  
                  // Başlık
                  const Text(
                    'Tehdit Durumu',
                    style: TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 1.2,
                    ),
                  ),
                  const SizedBox(height: 12),
                  
                  // Durum Kartları Grid
                  _buildStatusGrid(),
                  const SizedBox(height: 20),
                  
                  // Son Uyarı
                  if (_alertHistory.isNotEmpty) _buildLastAlertCard(),
                  
                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAppBar() {
    return SliverAppBar(
      expandedHeight: 120,
      pinned: true,
      backgroundColor: AppTheme.darkSurface,
      flexibleSpace: FlexibleSpaceBar(
        background: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Color(0xFF0F172A),
                Color(0xFF1E293B),
              ],
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 50, 20, 16),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [AppTheme.accentBlue, AppTheme.accentCyan],
                    ),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.shield_rounded, color: Colors.white, size: 24),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text(
                        'Sürücü Güvenlik',
                        style: TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 20,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      Row(
                        children: [
                          Container(
                            width: 7,
                            height: 7,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: _isConnected ? AppTheme.successGreen : AppTheme.dangerRed,
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            _isConnected ? 'Firebase Bağlı' : 'Bağlantı Yok',
                            style: TextStyle(
                              color: _isConnected ? AppTheme.successGreen : AppTheme.dangerRed,
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                // Bildirim sayısı badge
                if (_alertHistory.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: AppTheme.dangerRed.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: AppTheme.dangerRed.withOpacity(0.4)),
                    ),
                    child: Text(
                      '${_alertHistory.length}',
                      style: const TextStyle(
                        color: AppTheme.dangerRed,
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildActiveAlert() {
    final alertColor = _getAlertColor();
    return AnimatedBuilder(
      animation: _pulseAnim,
      builder: (context, child) {
        return Transform.scale(
          scale: _pulseAnim.value,
          child: AnimatedBuilder(
            animation: _glowAnim,
            builder: (context, child) {
              return Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      alertColor.withOpacity(0.2),
                      alertColor.withOpacity(0.05),
                    ],
                  ),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: alertColor.withOpacity(0.6 + _glowAnim.value * 0.4),
                    width: 1.5,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: alertColor.withOpacity(0.3 * _glowAnim.value),
                      blurRadius: 20,
                      spreadRadius: 2,
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    Icon(Icons.warning_amber_rounded, color: alertColor, size: 32),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            '⚠️ AKTİF UYARI',
                            style: TextStyle(
                              color: AppTheme.textSecondary,
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              letterSpacing: 1.2,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _currentAlert,
                            style: TextStyle(
                              color: alertColor,
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            },
          ),
        );
      },
    );
  }

  Widget _buildConnectionCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.darkCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.darkCardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.notifications_active_rounded,
                  color: AppTheme.accentBlue, size: 18),
              const SizedBox(width: 8),
              const Text(
                'Firebase Token (Python\'a kopyalayın)',
                style: TextStyle(
                  color: AppTheme.textSecondary,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          SelectableText(
            _fcmToken ?? 'Token yükleniyor...',
            style: const TextStyle(
              color: AppTheme.accentCyan,
              fontSize: 10,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusGrid() {
    final statuses = [
      StatusData(
        label: 'Göz Yorgunluğu',
        icon: Icons.remove_red_eye_rounded,
        isActive: _isTired,
        activeColor: AppTheme.dangerRed,
        activeLabel: 'TESPİT EDİLDİ',
        inactiveLabel: 'Normal',
      ),
      StatusData(
        label: 'Esneme',
        icon: Icons.sentiment_very_dissatisfied_rounded,
        isActive: _isYawning,
        activeColor: AppTheme.warningOrange,
        activeLabel: 'Esneme Var',
        inactiveLabel: 'Normal',
      ),
      StatusData(
        label: 'Baş Pozisyonu',
        icon: Icons.account_circle_rounded,
        isActive: _isHeadTilted,
        activeColor: AppTheme.warningOrange,
        activeLabel: 'Eğik!',
        inactiveLabel: 'Düzgün',
      ),
      StatusData(
        label: 'Yüz Görünümü',
        icon: Icons.face_rounded,
        isActive: _isFaceLost,
        activeColor: AppTheme.criticalRed,
        activeLabel: 'YÜZ YOK!',
        inactiveLabel: 'Görünür',
      ),
      StatusData(
        label: 'Sigara/Duman',
        icon: Icons.smoking_rooms_rounded,
        isActive: _isSmoking,
        activeColor: AppTheme.criticalRed,
        activeLabel: 'ALGILANDI',
        inactiveLabel: 'Yok',
      ),
      StatusData(
        label: 'Telefon',
        icon: Icons.phone_android_rounded,
        isActive: _isPhone,
        activeColor: AppTheme.phoneBlue,
        activeLabel: 'ALGILANDI',
        inactiveLabel: 'Yok',
      ),
    ];

    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 12,
        mainAxisSpacing: 12,
        childAspectRatio: 1.6,
      ),
      itemCount: statuses.length,
      itemBuilder: (context, index) => StatusCard(data: statuses[index]),
    );
  }

  Widget _buildLastAlertCard() {
    final last = _alertHistory.first;
    final time = last['time'] as DateTime;
    final timeStr =
        '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.darkCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.darkCardBorder),
      ),
      child: Row(
        children: [
          const Icon(Icons.access_time_rounded, color: AppTheme.textMuted, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Son Uyarı',
                  style: TextStyle(color: AppTheme.textMuted, fontSize: 11),
                ),
                Text(
                  last['message'],
                  style: const TextStyle(color: AppTheme.textPrimary, fontSize: 14),
                ),
              ],
            ),
          ),
          Text(
            timeStr,
            style: const TextStyle(color: AppTheme.textMuted, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

class StatusData {
  final String label;
  final IconData icon;
  final bool isActive;
  final Color activeColor;
  final String activeLabel;
  final String inactiveLabel;

  StatusData({
    required this.label,
    required this.icon,
    required this.isActive,
    required this.activeColor,
    required this.activeLabel,
    required this.inactiveLabel,
  });
}
