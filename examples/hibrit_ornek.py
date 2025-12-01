# ============================================
# 🔀 Hibrit Model Örneği
# ============================================
# 
# Bu dosya Hibrit öneri sisteminin
# detaylı kullanımını gösterir.
#
# Hibrit sistem, içerik tabanlı ve işbirlikçi
# filtrelemeyi birleştirerek daha iyi
# sonuçlar üretir.
#
# Çalıştırmak için:
# python examples/hibrit_ornek.py
# ============================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.models.hybrid import HybridModel
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeFilteringModel
from src.data_loader import veri_yukle
from src.preprocessing import egitim_test_ayir
from src.metrics import tum_metrikleri_hesapla, metrikleri_yazdir


def main():
    """
    🎯 Hibrit model detaylı kullanım örneği
    """
    
    print("=" * 60)
    print("🔀 HİBRİT ÖNERİ SİSTEMİ ÖRNEĞİ")
    print("=" * 60)
    
    # ----- 1. Veri Yükleme -----
    print("\n📂 1. Veri yükleniyor...")
    
    try:
        filmler, puanlar = veri_yukle()
    except FileNotFoundError:
        print("\n❌ Veri bulunamadı!")
        print("Önce veri setini indirin: python data/download_data.py")
        return
    
    # ----- 2. Hibrit Kavramı -----
    print("\n📚 2. Hibrit sistem nedir?")
    print("""
    ┌─────────────────────────────────────────────────────┐
    │              HİBRİT ÖNERİ SİSTEMİ                   │
    │                                                     │
    │   ┌───────────────┐     ┌───────────────────────┐   │
    │   │   İçerik      │     │    İşbirlikçi         │   │
    │   │   Tabanlı     │     │    Filtreleme         │   │
    │   │   Model       │     │    Modeli             │   │
    │   │               │     │                       │   │
    │   │  "Aksiyon     │     │  "Sana benzer         │   │
    │   │   seviyorsan  │     │   kullanıcılar        │   │
    │   │   aksiyon     │     │   şunları sevdi"      │   │
    │   │   önerelim"   │     │                       │   │
    │   └───────┬───────┘     └──────────┬────────────┘   │
    │           │                        │                │
    │           └──────────┬─────────────┘                │
    │                      ▼                              │
    │           ┌──────────────────────┐                  │
    │           │  Ağırlıklı Ortalama  │                  │
    │           │                      │                  │
    │           │  Tahmin = w1×T1 +    │                  │
    │           │           w2×T2      │                  │
    │           └──────────┬───────────┘                  │
    │                      ▼                              │
    │           ┌──────────────────────┐                  │
    │           │    FİNAL ÖNERİ       │                  │
    │           └──────────────────────┘                  │
    └─────────────────────────────────────────────────────┘
    """)
    
    # ----- 3. Modelleri Karşılaştır -----
    print("\n📊 3. Tekli modeller vs Hibrit karşılaştırması...")
    
    # Veriyi böl
    egitim, test = egitim_test_ayir(puanlar, test_orani=0.2)
    
    sonuclar = []
    
    # 3.1 İçerik Tabanlı
    print("\n   3.1 İçerik Tabanlı Model eğitiliyor...")
    model_icerik = ContentBasedModel()
    model_icerik.egit(filmler, egitim)
    
    # Test et
    gercek, tahminler = [], []
    for _, satir in test.head(500).iterrows():
        try:
            t = model_icerik.tahmin_et(satir['userId'], satir['movieId'])
            gercek.append(satir['rating'])
            tahminler.append(t)
        except Exception:
            continue
    
    metrikler = tum_metrikleri_hesapla(gercek, tahminler)
    sonuclar.append({'Model': 'İçerik Tabanlı', 'RMSE': metrikler['RMSE']})
    print(f"       RMSE: {metrikler['RMSE']:.4f}")
    
    # 3.2 İşbirlikçi
    print("\n   3.2 İşbirlikçi Model eğitiliyor...")
    model_isbirligi = CollaborativeFilteringModel(tur="kullanici")
    model_isbirligi.egit(filmler, egitim)
    
    gercek, tahminler = [], []
    for _, satir in test.head(500).iterrows():
        try:
            t = model_isbirligi.tahmin_et(satir['userId'], satir['movieId'])
            gercek.append(satir['rating'])
            tahminler.append(t)
        except Exception:
            continue
    
    metrikler = tum_metrikleri_hesapla(gercek, tahminler)
    sonuclar.append({'Model': 'İşbirlikçi', 'RMSE': metrikler['RMSE']})
    print(f"       RMSE: {metrikler['RMSE']:.4f}")
    
    # 3.3 Hibrit
    print("\n   3.3 Hibrit Model eğitiliyor...")
    model_hibrit = HybridModel(icerik_agirligi=0.3, isbirligi_agirligi=0.7)
    model_hibrit.egit(filmler, egitim)
    
    gercek, tahminler = [], []
    for _, satir in test.head(500).iterrows():
        try:
            t = model_hibrit.tahmin_et(satir['userId'], satir['movieId'])
            gercek.append(satir['rating'])
            tahminler.append(t)
        except Exception:
            continue
    
    metrikler = tum_metrikleri_hesapla(gercek, tahminler)
    sonuclar.append({'Model': 'Hibrit', 'RMSE': metrikler['RMSE']})
    print(f"       RMSE: {metrikler['RMSE']:.4f}")
    
    # Sonuç tablosu
    print("\n   📊 Karşılaştırma Tablosu:")
    print("   " + "-" * 35)
    print(f"   {'Model':<20} | {'RMSE':<10}")
    print("   " + "-" * 35)
    for s in sorted(sonuclar, key=lambda x: x['RMSE']):
        print(f"   {s['Model']:<20} | {s['RMSE']:.4f}")
    print("   " + "-" * 35)
    
    # ----- 4. Cold Start Senaryosu -----
    print("\n❄️ 4. Cold Start senaryosu...")
    print("""
    Cold Start nedir?
    Yeni kullanıcı veya yeni film = Çok az veri
    
    Problem:
    - İşbirlikçi filtreleme çalışamaz (benzer yok!)
    - Sadece içerik tabanlı kullanılabilir
    
    Hibrit çözümü:
    - Az veri → İçerik ağırlığı artır
    - Çok veri → İşbirlikçi ağırlığı artır
    """)
    
    # Örnek kullanıcılar
    test_kullanicilar = [1, 100, 200]
    
    for uid in test_kullanicilar:
        durum = model_hibrit.cold_start_tespit_et(uid)
        print(f"   Kullanıcı {uid}: {durum}")
    
    # ----- 5. Öneri Analizi -----
    print("\n🔍 5. Öneri kaynağı analizi...")
    
    # Bir kullanıcı için öneri al
    ornek_kullanici = egitim['userId'].iloc[0]
    oneriler = model_hibrit.oner(ornek_kullanici, n=3)
    
    print(f"\n   Kullanıcı {ornek_kullanici} için öneriler analizi:")
    
    for film_id, tahmin in oneriler:
        analiz = model_hibrit.onerinin_kaynagini_analiz_et(ornek_kullanici, film_id)
        
        print(f"\n   🎬 {analiz['film_adi'][:40]}")
        print(f"      İçerik tahmini: {analiz['icerik_tahmin']:.2f}")
        print(f"      İşbirlikçi tahmini: {analiz['isbirligi_tahmin']:.2f}")
        print(f"      Hibrit tahmin: {analiz['hibrit_tahmin']:.2f}")
        
        # Hangi model daha etkili?
        if analiz['icerik_katki'] > analiz['isbirligi_katki']:
            print(f"      → İçerik tabanlı daha etkili")
        else:
            print(f"      → İşbirlikçi daha etkili")
    
    # ----- 6. Ağırlık Ayarlama -----
    print("\n⚖️ 6. Ağırlık ayarlama stratejileri...")
    print("""
    Farklı ağırlık kombinasyonları:
    
    1. İçerik ağırlıklı (0.7, 0.3):
       - Yeni kullanıcılar için iyi
       - Sürpriz keşif az
    
    2. İşbirlikçi ağırlıklı (0.3, 0.7):
       - Aktif kullanıcılar için iyi
       - Sürpriz keşifler mümkün
    
    3. Dengeli (0.5, 0.5):
       - Genel amaçlı
       - Her durumda orta performans
    
    4. Dinamik (kullanıcıya göre değişen):
       - Cold start için içerik ağırlıklı
       - Aktif kullanıcı için işbirlikçi ağırlıklı
    """)
    
    # ----- Sonuç -----
    print("\n" + "=" * 60)
    print("✅ HİBRİT ÖRNEĞİ TAMAMLANDI!")
    print("=" * 60)
    
    print("""
    📚 Öğrendiklerimiz:
    
    1. Hibrit sistem birden fazla modeli birleştirir
    2. Ağırlıklı ortalama ile tahminler birleşir
    3. Cold Start problemini azaltır
    4. Daha dengeli öneriler üretir
    
    🎯 Hibrit'in Avantajları:
    - Her iki yöntemin güçlü yanlarını kullanır
    - Cold Start'a daha dirençli
    - Farklı senaryolara uyum sağlar
    
    ⚠️ Hibrit'in Dezavantajları:
    - Daha karmaşık sistem
    - Ağırlık ayarı gerektirir
    - Hesaplama maliyeti daha yüksek
    
    💡 İpucu: Gerçek uygulamalarda dinamik ağırlıklar kullanın!
    """)


if __name__ == "__main__":
    main()
