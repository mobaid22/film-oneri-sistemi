#!/usr/bin/env python3
# ============================================
# 🎬 Film Öneri Sistemi - Demo Uygulaması
# ============================================
# 
# Bu dosya basit bir komut satırı (CLI) arayüzü
# sağlar. Sistemi interaktif olarak kullanabilirsiniz.
#
# Çalıştırmak için:
# python app.py
# ============================================

import sys
import os

# src klasörünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def banner_goster():
    """Hoş geldin banner'ı gösterir."""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🎬 FİLM ÖNERİ SİSTEMİ                                 ║
║                                                          ║
║   Türkçe açıklamalı, başlangıç seviyesi                 ║
║   makine öğrenmesi projesi                               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


def menu_goster():
    """Ana menüyü gösterir."""
    print("""
┌──────────────────────────────────────┐
│            ANA MENÜ                  │
├──────────────────────────────────────┤
│                                      │
│  1. 📥 Veri setini indir             │
│  2. 📂 Veri setini yükle             │
│  3. 🔧 Model eğit                    │
│  4. 🎬 Film önerisi al               │
│  5. 🔍 Film ara                      │
│  6. 🏆 Popüler filmleri gör          │
│  7. 👤 Kullanıcı profili gör         │
│  8. 📊 Model performansını gör       │
│  9. 📈 Tüm modelleri karşılaştır     │
│  0. 🚪 Çıkış                         │
│                                      │
└──────────────────────────────────────┘
    """)


def veri_indir():
    """Veri setini indirir."""
    print("\n📥 Veri seti indiriliyor...")
    try:
        from data.download_data import ana_fonksiyon
        ana_fonksiyon()
    except Exception as e:
        print(f"❌ Hata: {e}")


def veri_yukle(sistem):
    """Veri setini yükler."""
    try:
        sistem.veri_yukle()
        return True
    except Exception as e:
        print(f"❌ Hata: {e}")
        print("\nÖnce '1' seçeneği ile veri setini indirin!")
        return False


def model_egit(sistem):
    """Model eğitir."""
    if not sistem.veri_yuklendi:
        print("❌ Önce veri setini yükleyin! (Seçenek 2)")
        return
    
    print("""
    Model Seçenekleri:
    ─────────────────
    1. icerik    - İçerik Tabanlı Filtreleme
    2. kullanici - Kullanıcı Tabanlı İşbirlikçi
    3. film      - Film Tabanlı İşbirlikçi
    4. svd       - SVD Matris Ayrıştırma
    5. hibrit    - Hibrit Sistem (Önerilen)
    """)
    
    secim = input("Model seçin (1-5): ").strip()
    
    model_map = {
        '1': 'icerik',
        '2': 'kullanici',
        '3': 'film',
        '4': 'svd',
        '5': 'hibrit'
    }
    
    if secim in model_map:
        sistem.model_egit(model_map[secim])
    else:
        print("❌ Geçersiz seçim!")


def film_oner(sistem):
    """Film önerisi alır."""
    if sistem.aktif_model is None:
        print("❌ Önce model eğitin! (Seçenek 3)")
        return
    
    try:
        kullanici_id = int(input("\nKullanıcı ID girin: ").strip())
        n = int(input("Kaç film önerilsin? (varsayılan 10): ").strip() or "10")
        
        sistem.film_oner(kullanici_id, n=n)
        
    except ValueError:
        print("❌ Geçersiz giriş! Sayı girin.")


def film_ara(sistem):
    """Film arar."""
    if not sistem.veri_yuklendi:
        print("❌ Önce veri setini yükleyin!")
        return
    
    arama = input("\nFilm adı girin: ").strip()
    if arama:
        sistem.film_ara(arama)


def populer_filmler(sistem):
    """Popüler filmleri gösterir."""
    if not sistem.veri_yuklendi:
        print("❌ Önce veri setini yükleyin!")
        return
    
    sistem.populer_filmler(limit=10)


def kullanici_profili(sistem):
    """Kullanıcı profilini gösterir."""
    if not sistem.veri_yuklendi:
        print("❌ Önce veri setini yükleyin!")
        return
    
    try:
        kullanici_id = int(input("\nKullanıcı ID girin: ").strip())
        sistem.kullanici_profili(kullanici_id)
    except ValueError:
        print("❌ Geçersiz giriş!")


def model_performans(sistem):
    """Model performansını gösterir."""
    if sistem.aktif_model is None:
        print("❌ Önce model eğitin!")
        return
    
    sistem.modeli_degerlendir()


def modelleri_karsilastir(sistem):
    """Tüm modelleri karşılaştırır."""
    if not sistem.veri_yuklendi:
        print("❌ Önce veri setini yükleyin!")
        return
    
    print("\n⚠️ Bu işlem uzun sürebilir...")
    onay = input("Devam etmek istiyor musunuz? (e/h): ").strip().lower()
    
    if onay == 'e':
        sistem.tum_modelleri_karsilastir()


def main():
    """Ana program döngüsü."""
    
    banner_goster()
    
    # Sistemi oluştur
    from src.recommender import FilmOneriSistemi
    sistem = FilmOneriSistemi()
    
    while True:
        menu_goster()
        
        secim = input("Seçiminiz (0-9): ").strip()
        
        if secim == '0':
            print("\n👋 Güle güle! Tekrar görüşmek üzere.")
            break
        elif secim == '1':
            veri_indir()
        elif secim == '2':
            veri_yukle(sistem)
        elif secim == '3':
            model_egit(sistem)
        elif secim == '4':
            film_oner(sistem)
        elif secim == '5':
            film_ara(sistem)
        elif secim == '6':
            populer_filmler(sistem)
        elif secim == '7':
            kullanici_profili(sistem)
        elif secim == '8':
            model_performans(sistem)
        elif secim == '9':
            modelleri_karsilastir(sistem)
        else:
            print("❌ Geçersiz seçim! 0-9 arası bir sayı girin.")
        
        input("\nDevam etmek için Enter'a basın...")


if __name__ == "__main__":
    main()
