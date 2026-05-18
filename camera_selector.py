"""
Kamera Seçim Modülü
===================
Program başlangıcında 3 seçenek sunar:
  1. Laptop Kamerası      (dahili webcam, index 0)
  2. iPhone – Wi-Fi       (IP Camera uygulaması, MJPEG stream URL)
  3. iPhone – USB         (Camo / EpocCam sanal sürücü, kamera index 1/2/...)

iPhone Wi-Fi için App Store'dan ücretsiz:
  "IP Camera Lite" veya "IP Webcam"  →  http://<IP>:<PORT>/video

iPhone USB için PC'ye kurulması gereken (ücretsiz sürüm mevcut):
  Camo (reincubate.com/camo)  ya da  EpocCam (Elgato)
  Kurulumdan sonra iPhone sanal bir webcam olarak görünür.
"""

import tkinter as tk
from tkinter import messagebox
import json
import os
import cv2
import threading

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")

STREAM_PATHS = ["/video", "/videofeed", "/mjpeg", "/shot.jpg", ""]

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    defaults = {
        "last_source": "laptop",
        "iphone_ip": "",
        "iphone_port": "8080",
        "usb_index": "1",
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                defaults.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return defaults


def save_config(data: dict):
    try:
        existing = load_config()
        existing.update(data)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[UYARI] Config kaydedilemedi: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Bağlantı testi yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def test_wifi_stream(ip: str, port: str) -> tuple[bool, str]:
    """Çalışan Wi-Fi stream URL'sini döndürür."""
    for path in STREAM_PATHS:
        url = f"http://{ip}:{port}{path}"
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return True, url
        cap.release()
    return False, ""


def test_usb_index(index: int) -> bool:
    """Verilen kamera indeksinin açılabildiğini test eder."""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, _ = cap.read()
        cap.release()
        return ret
    cap.release()
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Ana GUI
# ─────────────────────────────────────────────────────────────────────────────

class CameraSelector:
    BG      = "#0f0f1a"
    CARD    = "#1a1a2e"
    CARD_SEL= "#1e1e40"
    ACCENT  = "#6c63ff"
    TEXT    = "#e0e0f0"
    SUBTEXT = "#8888aa"
    SUCCESS = "#4ade80"
    ERROR   = "#f87171"
    WARN    = "#facc15"
    BORDER  = "#2a2a45"
    HIGHLIGHT = "#3a3a6e"

    def __init__(self):
        self.result = None
        self._test_thread = None
        self._confirmed_wifi_url = None

        self.root = tk.Tk()
        self.root.title("Kamera Kaynağı Seç")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG)

        w, h = 560, 560
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.config = load_config()
        self._mode = "laptop"
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Başlık
        hdr = tk.Frame(self.root, bg=self.BG)
        hdr.pack(fill="x", padx=28, pady=(24, 0))
        tk.Label(hdr, text="🎥  Kamera Kaynağı Seç",
                 font=("Segoe UI", 17, "bold"), bg=self.BG, fg=self.TEXT).pack(anchor="w")
        tk.Label(hdr, text="Analiz için hangi kamerayı kullanmak istersiniz?",
                 font=("Segoe UI", 9), bg=self.BG, fg=self.SUBTEXT).pack(anchor="w", pady=(3, 0))

        tk.Frame(self.root, bg=self.BORDER, height=1).pack(fill="x", padx=28, pady=14)

        # Kartlar
        cards_row = tk.Frame(self.root, bg=self.BG)
        cards_row.pack(fill="x", padx=28)

        self._cards = {}
        card_defs = [
            ("laptop",   "💻", "Laptop Kamerası",   "Dahili webcam\nKurulum gerektirmez"),
            ("wifi",     "📶", "iPhone – Wi-Fi",    "IP Camera uygulaması\nAynı Wi-Fi ağı gerekli"),
            ("usb",      "🔌", "iPhone – USB",      "Camo / EpocCam\nSanal sürücü ile bağlantı"),
        ]
        for key, emoji, title, desc in card_defs:
            card = self._make_card(cards_row, emoji, title, desc, key)
            self._cards[key] = card
            card.pack(side="left", expand=True, fill="both", padx=(0, 8 if key != "usb" else 0))

        # Ayar panelleri (başta hepsi gizli)
        self._wifi_panel = self._build_wifi_panel()
        self._usb_panel  = self._build_usb_panel()

        # Durum satırı
        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(self.root, textvariable=self._status_var,
                                    font=("Segoe UI", 9), bg=self.BG, fg=self.SUBTEXT,
                                    wraplength=510, justify="left")
        self._status_lbl.pack(anchor="w", padx=28, pady=(4, 0))

        # Başlat butonu
        self._start_btn = tk.Button(
            self.root, text="▶  Başlat",
            font=("Segoe UI", 11, "bold"),
            bg=self.ACCENT, fg="white",
            activebackground="#5a52d5", activeforeground="white",
            relief="flat", bd=0, cursor="hand2",
            padx=24, pady=11,
            command=self._start,
        )
        self._start_btn.pack(fill="x", padx=28, pady=(10, 24))

        # Başlangıç seçimi
        last = self.config.get("last_source", "laptop")
        if last in self._cards:
            self._select(last)
        else:
            self._select("laptop")

    # ── Kart factory ──────────────────────────────────────────────────

    def _make_card(self, parent, emoji, title, desc, key):
        frame = tk.Frame(parent, bg=self.CARD, cursor="hand2", relief="flat")
        inner = tk.Frame(frame, bg=self.CARD, padx=10, pady=14)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=emoji, font=("Segoe UI Emoji", 24),
                 bg=self.CARD, fg=self.TEXT).pack()
        tk.Label(inner, text=title, font=("Segoe UI", 10, "bold"),
                 bg=self.CARD, fg=self.TEXT).pack(pady=(5, 0))
        tk.Label(inner, text=desc, font=("Segoe UI", 8),
                 bg=self.CARD, fg=self.SUBTEXT,
                 wraplength=140, justify="center").pack(pady=(3, 0))

        for w in [frame, inner] + inner.winfo_children():
            w.bind("<Button-1>", lambda e, k=key: self._select(k))
        return frame

    # ── Wi-Fi paneli ──────────────────────────────────────────────────

    def _build_wifi_panel(self):
        panel = tk.Frame(self.root, bg=self.BG)

        tk.Label(panel, text="Wi-Fi Stream Ayarları",
                 font=("Segoe UI", 10, "bold"), bg=self.BG, fg=self.TEXT
                 ).grid(row=0, column=0, columnspan=5, sticky="w", padx=28, pady=(12, 6))

        tk.Label(panel, text="IP Adresi:", font=("Segoe UI", 9),
                 bg=self.BG, fg=self.SUBTEXT).grid(row=1, column=0, sticky="w", padx=(28, 4))

        self._ip_var = tk.StringVar(value=self.config.get("iphone_ip", ""))
        ip_e = tk.Entry(panel, textvariable=self._ip_var, font=("Consolas", 10),
                        bg=self.CARD, fg=self.TEXT, insertbackground=self.TEXT,
                        relief="flat", bd=0, width=18)
        ip_e.grid(row=1, column=1, ipady=5, padx=(0, 8))

        tk.Label(panel, text="Port:", font=("Segoe UI", 9),
                 bg=self.BG, fg=self.SUBTEXT).grid(row=1, column=2, sticky="w")

        self._port_var = tk.StringVar(value=self.config.get("iphone_port", "8080"))
        tk.Entry(panel, textvariable=self._port_var, font=("Consolas", 10),
                 bg=self.CARD, fg=self.TEXT, insertbackground=self.TEXT,
                 relief="flat", bd=0, width=6).grid(row=1, column=3, ipady=5, padx=(4, 8))

        self._wifi_test_btn = tk.Button(
            panel, text="🔗 Test Et",
            font=("Segoe UI", 9), bg=self.BORDER, fg=self.TEXT,
            activebackground=self.ACCENT, activeforeground="white",
            relief="flat", bd=0, cursor="hand2", padx=8, pady=4,
            command=self._test_wifi,
        )
        self._wifi_test_btn.grid(row=1, column=4, padx=(0, 28))

        hint = ("💡 iPhone'da 'IP Camera Lite' veya 'IP Webcam' uygulamasını açın. "
                "Uygulamanın gösterdiği IP ve portu girin. "
                "PC ve iPhone aynı Wi-Fi'de olmalı.")
        tk.Label(panel, text=hint, font=("Segoe UI", 8), bg=self.BG, fg=self.SUBTEXT,
                 wraplength=500, justify="left"
                 ).grid(row=2, column=0, columnspan=5, sticky="w", padx=28, pady=(8, 0))

        return panel

    # ── USB paneli ────────────────────────────────────────────────────

    def _build_usb_panel(self):
        panel = tk.Frame(self.root, bg=self.BG)

        tk.Label(panel, text="USB Sanal Webcam Ayarları",
                 font=("Segoe UI", 10, "bold"), bg=self.BG, fg=self.TEXT
                 ).grid(row=0, column=0, columnspan=4, sticky="w", padx=28, pady=(12, 6))

        tk.Label(panel, text="Kamera İndeksi:", font=("Segoe UI", 9),
                 bg=self.BG, fg=self.SUBTEXT).grid(row=1, column=0, sticky="w", padx=(28, 4))

        self._usb_index_var = tk.StringVar(value=self.config.get("usb_index", "1"))
        tk.Entry(panel, textvariable=self._usb_index_var, font=("Consolas", 10),
                 bg=self.CARD, fg=self.TEXT, insertbackground=self.TEXT,
                 relief="flat", bd=0, width=4
                 ).grid(row=1, column=1, ipady=5, padx=(0, 8))

        self._usb_test_btn = tk.Button(
            panel, text="🔗 Test Et",
            font=("Segoe UI", 9), bg=self.BORDER, fg=self.TEXT,
            activebackground=self.ACCENT, activeforeground="white",
            relief="flat", bd=0, cursor="hand2", padx=8, pady=4,
            command=self._test_usb,
        )
        self._usb_test_btn.grid(row=1, column=2, padx=(0, 28))

        hint = ("💡 Camo veya EpocCam kurulduktan sonra iPhone'u USB ile bağlayın.\n"
                "   Dahili kamera genellikle 0'dır; iPhone sanal kamera 1 veya 2 olur.\n"
                "   Doğru indeksi bulmak için 'Test Et' butonunu kullanın.")
        tk.Label(panel, text=hint, font=("Segoe UI", 8), bg=self.BG, fg=self.SUBTEXT,
                 wraplength=500, justify="left"
                 ).grid(row=2, column=0, columnspan=4, sticky="w", padx=28, pady=(8, 0))

        return panel

    # ──────────────────────────────────────────────────────────────────
    # Seçim
    # ──────────────────────────────────────────────────────────────────

    def _select(self, key: str):
        self._mode = key
        self._confirmed_wifi_url = None

        # Kart renklerini güncelle
        for k, card in self._cards.items():
            bg = self.HIGHLIGHT if k == key else self.CARD
            self._set_bg_recursive(card, bg)

        # Panel görünürlüğü
        self._wifi_panel.pack_forget()
        self._usb_panel.pack_forget()

        if key == "wifi":
            self._wifi_panel.pack(fill="x", before=self._status_lbl)
            self._status_var.set("IP adresini girin ve bağlantıyı test edin.")
            self._status_lbl.configure(fg=self.SUBTEXT)
        elif key == "usb":
            self._usb_panel.pack(fill="x", before=self._status_lbl)
            self._status_var.set("Kamera indeksini girin (varsayılan: 1) ve test edin.")
            self._status_lbl.configure(fg=self.SUBTEXT)
        else:
            self._status_var.set("")

    def _set_bg_recursive(self, widget, color):
        try:
            widget.configure(bg=color)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_bg_recursive(child, color)

    # ──────────────────────────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────────────────────────

    def _run_test(self, btn, label, fn):
        """Generic test runner (thread)."""
        if self._test_thread and self._test_thread.is_alive():
            return
        self._status_var.set("⏳  Test ediliyor...")
        self._status_lbl.configure(fg=self.SUBTEXT)
        btn.configure(state="disabled", text="⏳ Test...")

        def run():
            result = fn()
            self.root.after(0, lambda: label(result, btn))

        self._test_thread = threading.Thread(target=run, daemon=True)
        self._test_thread.start()

    # Wi-Fi testi
    def _test_wifi(self):
        def task():
            ip   = self._ip_var.get().strip()
            port = self._port_var.get().strip()
            return test_wifi_stream(ip, port)

        def on_done(result, btn):
            ok, url = result
            btn.configure(state="normal", text="🔗 Test Et")
            if ok:
                self._confirmed_wifi_url = url
                self._status_var.set(f"✅  Bağlantı başarılı!  →  {url}")
                self._status_lbl.configure(fg=self.SUCCESS)
            else:
                self._confirmed_wifi_url = None
                self._status_var.set("❌  Bağlanamadı. IP/Port ve Wi-Fi bağlantısını kontrol edin.")
                self._status_lbl.configure(fg=self.ERROR)

        self._run_test(self._wifi_test_btn, on_done, task)

    # USB testi
    def _test_usb(self):
        def task():
            try:
                idx = int(self._usb_index_var.get().strip())
            except ValueError:
                return None
            return (idx, test_usb_index(idx))

        def on_done(result, btn):
            btn.configure(state="normal", text="🔗 Test Et")
            if result is None:
                self._status_var.set("❌  Geçersiz kamera indeksi. Sayısal bir değer girin.")
                self._status_lbl.configure(fg=self.ERROR)
                return
            idx, ok = result
            if ok:
                self._status_var.set(f"✅  Kamera {idx} bulundu ve çalışıyor!")
                self._status_lbl.configure(fg=self.SUCCESS)
            else:
                self._status_var.set(
                    f"❌  Kamera {idx} bulunamadı. "
                    "Camo/EpocCam kurulu ve iPhone bağlı mı? Farklı indeks deneyin (1, 2, …)."
                )
                self._status_lbl.configure(fg=self.ERROR)

        self._run_test(self._usb_test_btn, on_done, task)

    # ──────────────────────────────────────────────────────────────────
    # Başlat
    # ──────────────────────────────────────────────────────────────────

    def _start(self):
        if self._mode == "laptop":
            save_config({"last_source": "laptop"})
            self.result = ("laptop", 0)
            self.root.destroy()

        elif self._mode == "wifi":
            ip   = self._ip_var.get().strip()
            port = self._port_var.get().strip()
            if not ip:
                messagebox.showwarning("Uyarı", "Lütfen iPhone IP adresini girin.", parent=self.root)
                return
            if self._confirmed_wifi_url is None:
                url = f"http://{ip}:{port}/video"
                if not messagebox.askyesno(
                    "Test Edilmedi",
                    f"Bağlantı test edilmedi.\n\nURL: {url}\n\nYine de devam edilsin mi?",
                    parent=self.root
                ):
                    return
                self._confirmed_wifi_url = url
            save_config({"last_source": "wifi", "iphone_ip": ip, "iphone_port": port})
            self.result = ("wifi", self._confirmed_wifi_url)
            self.root.destroy()

        elif self._mode == "usb":
            try:
                idx = int(self._usb_index_var.get().strip())
            except ValueError:
                messagebox.showwarning("Uyarı", "Geçerli bir kamera indeksi girin.", parent=self.root)
                return
            save_config({"last_source": "usb", "usb_index": str(idx)})
            self.result = ("usb", idx)
            self.root.destroy()

    def _on_close(self):
        self.result = None
        self.root.destroy()

    # ──────────────────────────────────────────────────────────────────
    # Çalıştır
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()
        return self.result


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def select_camera():
    """
    Kamera seçim penceresini gösterir.

    Dönen değer:
      ("laptop", 0)             → dahili kamera (index 0)
      ("wifi",   "http://...")  → iPhone Wi-Fi stream URL
      ("usb",    1)             → iPhone USB sanal webcam (index 1/2/…)
      None                      → pencere kapatıldı → programı sonlandır
    """
    return CameraSelector().run()


if __name__ == "__main__":
    print("Seçim:", select_camera())
