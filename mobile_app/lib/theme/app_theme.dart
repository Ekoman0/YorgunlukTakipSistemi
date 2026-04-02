import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Renkler
  static const Color darkBackground = Color(0xFF0A0E1A);
  static const Color darkSurface = Color(0xFF111827);
  static const Color darkCard = Color(0xFF1A2235);
  static const Color darkCardBorder = Color(0xFF243050);

  static const Color accentBlue = Color(0xFF3B82F6);
  static const Color accentCyan = Color(0xFF06B6D4);
  static const Color successGreen = Color(0xFF10B981);
  static const Color warningOrange = Color(0xFFF59E0B);
  static const Color dangerRed = Color(0xFFEF4444);
  static const Color criticalRed = Color(0xFFDC2626);
  static const Color phoneBlue = Color(0xFF6366F1);

  static const Color textPrimary = Color(0xFFF1F5F9);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF475569);

  static ThemeData get darkTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: darkBackground,
      colorScheme: const ColorScheme.dark(
        primary: accentBlue,
        secondary: accentCyan,
        surface: darkSurface,
        error: dangerRed,
      ),
      textTheme: TextTheme(
        displayLarge: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontWeight: FontWeight.w700,
        ),
        displayMedium: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontWeight: FontWeight.w600,
        ),
        headlineLarge: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontWeight: FontWeight.w700,
          fontSize: 24,
        ),
        headlineMedium: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontWeight: FontWeight.w600,
          fontSize: 20,
        ),
        titleLarge: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontWeight: FontWeight.w600,
          fontSize: 18,
        ),
        bodyLarge: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontSize: 16,
        ),
        bodyMedium: const TextStyle(
          fontFamily: 'Inter',
          color: textSecondary,
          fontSize: 14,
        ),
        labelLarge: const TextStyle(
          fontFamily: 'Inter',
          color: textPrimary,
          fontWeight: FontWeight.w600,
          fontSize: 14,
        ),
      ),
      cardTheme: CardTheme(
        color: darkCard,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: const BorderSide(color: darkCardBorder, width: 1),
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        backgroundColor: darkSurface,
        selectedItemColor: accentBlue,
        unselectedItemColor: textMuted,
        elevation: 0,
      ),
      dividerColor: darkCardBorder,
      iconTheme: const IconThemeData(color: textSecondary),
    );
  }
}
