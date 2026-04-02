import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../main.dart';

class AlertHistoryScreen extends StatefulWidget {
  const AlertHistoryScreen({super.key});

  @override
  State<AlertHistoryScreen> createState() => _AlertHistoryScreenState();
}

class _AlertHistoryScreenState extends State<AlertHistoryScreen> {
  final List<Map<String, dynamic>> _history = [];

  @override
  void initState() {
    super.initState();
    AlertEventBus.instance.addListener(_onAlert);
  }

  void _onAlert(Map<String, dynamic> data) {
    setState(() {
      _history.insert(0, {
        'time': DateTime.now(),
        'message': data['message'] ?? '',
        'type': data['alert_type'] ?? 'general',
      });
    });
  }

  @override
  void dispose() {
    AlertEventBus.instance.removeListener(_onAlert);
    super.dispose();
  }

  Color _typeColor(String type) {
    switch (type) {
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

  IconData _typeIcon(String type) {
    switch (type) {
      case 'tired':
        return Icons.remove_red_eye_rounded;
      case 'face_lost':
        return Icons.face_retouching_off_rounded;
      case 'smoking':
        return Icons.smoking_rooms_rounded;
      case 'phone':
        return Icons.phone_android_rounded;
      case 'yawning':
        return Icons.sentiment_very_dissatisfied_rounded;
      case 'head_tilt':
        return Icons.rotate_90_degrees_ccw_rounded;
      default:
        return Icons.warning_amber_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.darkBackground,
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            pinned: true,
            backgroundColor: AppTheme.darkSurface,
            title: Row(
              children: [
                const Text(
                  'Uyarı Geçmişi',
                  style: TextStyle(
                    color: AppTheme.textPrimary,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (_history.isNotEmpty) ...[
                  const SizedBox(width: 10),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: AppTheme.dangerRed.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                          color: AppTheme.dangerRed.withOpacity(0.4)),
                    ),
                    child: Text(
                      '${_history.length}',
                      style: const TextStyle(
                        color: AppTheme.dangerRed,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ],
              ],
            ),
            actions: [
              if (_history.isNotEmpty)
                IconButton(
                  icon: const Icon(Icons.delete_sweep_rounded,
                      color: AppTheme.textSecondary),
                  onPressed: () => setState(() => _history.clear()),
                  tooltip: 'Geçmişi Temizle',
                ),
            ],
          ),
          if (_history.isEmpty)
            SliverFillRemaining(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: AppTheme.darkCard,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.check_circle_outline_rounded,
                        color: AppTheme.successGreen,
                        size: 48,
                      ),
                    ),
                    const SizedBox(height: 20),
                    const Text(
                      'Henüz Uyarı Yok',
                      style: TextStyle(
                        color: AppTheme.textPrimary,
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Sistem aktif uyarıları burada gösterecek',
                      style: TextStyle(
                        color: AppTheme.textMuted,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            )
          else
            SliverPadding(
              padding: const EdgeInsets.all(16),
              sliver: SliverList(
                delegate: SliverChildBuilderDelegate(
                  (context, index) {
                    final item = _history[index];
                    final time = item['time'] as DateTime;
                    final type = item['type'] as String;
                    final color = _typeColor(type);
                    final timeStr =
                        '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}:${time.second.toString().padLeft(2, '0')}';
                    final dateStr =
                        '${time.day.toString().padLeft(2, '0')}.${time.month.toString().padLeft(2, '0')}.${time.year}';

                    return Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: AppTheme.darkCard,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: index == 0
                                ? color.withOpacity(0.5)
                                : AppTheme.darkCardBorder,
                          ),
                        ),
                        child: Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(10),
                              decoration: BoxDecoration(
                                color: color.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Icon(_typeIcon(type), color: color, size: 20),
                            ),
                            const SizedBox(width: 14),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    item['message'],
                                    style: const TextStyle(
                                      color: AppTheme.textPrimary,
                                      fontWeight: FontWeight.w600,
                                      fontSize: 14,
                                    ),
                                  ),
                                  const SizedBox(height: 2),
                                  Text(
                                    '$dateStr  $timeStr',
                                    style: const TextStyle(
                                      color: AppTheme.textMuted,
                                      fontSize: 11,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            if (index == 0)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(
                                  color: color.withOpacity(0.1),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: color.withOpacity(0.3)),
                                ),
                                child: Text(
                                  'YENİ',
                                  style: TextStyle(
                                    color: color,
                                    fontSize: 10,
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    );
                  },
                  childCount: _history.length,
                ),
              ),
            ),
        ],
      ),
    );
  }
}
