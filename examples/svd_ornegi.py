# ============================================
# 🔢 SVD Model Örneği
# ============================================
# 
# Bu dosya SVD (Singular Value Decomposition)
# modelinin detaylı kullanımını gösterir.
#
# Çalıştırmak için:
# python examples/svd_ornegi.py
# ============================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.models.svd_model import SVDModel
from src.data_loader import veri_yukle
from src.preprocessing import egitim_test_ayir
from src.metrics import tum_metrikleri_hesapla, metrikleri_yazdir


def main():
    """
    🎯 SVD modeli detaylı kullanım örneği
    """
    
    print("=" * 60)
    print("🔢 SVD (Singular Value Decomposition) ÖRNEĞİ")
    print("=" * 60)
    
    # ----- 1. Veri Yükleme -----
    print("\n📂 1. Veri yükleniyor...")
    
    try:
        filmler, puanlar = veri_yukle()
    except FileNotFoundError:
        print("\n❌ Veri bulunamadı!")
        print("Önce veri setini indirin: python data/download_data.py")
        return
    
    # ----- 2. Veri Bölme -----
    print("\n✂️ 2. Veri eğitim/test olarak bölünüyor...")
    egitim, test = egitim_test_ayir(puanlar, test_orani=0.2)
    
    # ----- 3. Model Parametrelerini Ayarla -----
    print("\n🔧 3. SVD parametreleri ayarlanıyor...")
    
    # Hiperparametreler ve açıklamaları
    parametreler = {
        'n_faktor': 50,      # Gizli faktör sayısı
        'ogrenme_orani': 0.005,  # Gradient descent adım boyutu
        'regulasyon': 0.02,  # Overfitting önleme katsayısı
        'iterasyon': 20      # Eğitim epoch sayısı
    }
    
    print(f"""
    SVD Hiperparametreleri:
    
    n_faktor = {parametreler['n_faktor']}
    ─────────
    Gizli faktör sayısı. Her kullanıcı ve film
    bu kadar boyutlu bir vektörle temsil edilir.
    
    Yüksek değer → Daha karmaşık kalıplar yakalanır
    Düşük değer → Daha hızlı eğitim, daha az overfitting
    Önerilen: 20-100 arası
    
    ogrenme_orani = {parametreler['ogrenme_orani']}
    ─────────
    Her güncelleme adımının büyüklüğü.
    
    Yüksek değer → Hızlı ama kararsız öğrenme
    Düşük değer → Yavaş ama kararlı öğrenme
    Önerilen: 0.001-0.01 arası
    
    regulasyon = {parametreler['regulasyon']}
    ─────────
    Overfitting (aşırı öğrenme) önleme katsayısı.
    
    Yüksek değer → Daha basit model, daha iyi genelleme
    Düşük değer → Veriye daha yakın model
    Önerilen: 0.01-0.1 arası
    
    iterasyon = {parametreler['iterasyon']}
    ─────────
    Tüm veri üzerinden kaç kez geçilecek.
    
    Fazla iterasyon → Daha iyi öğrenme ama overfitting riski
    Az iterasyon → Yetersiz öğrenme (underfitting)
    Önerilen: 10-50 arası
    """)
    
    # ----- 4. Model Eğitimi -----
    print("\n🔄 4. SVD modeli eğitiliyor...")
    
    model = SVDModel(**parametreler)
    model.egit(filmler, egitim)
    
    # ----- 5. Model Değerlendirme -----
    print("\n📊 5. Model değerlendiriliyor...")
    
    # Test seti üzerinde tahmin yap
    gercek = []
    tahminler = []
    
    print(f"   {len(test)} test puanı için tahmin yapılıyor...")
    
    for _, satir in test.iterrows():
        try:
            tahmin = model.tahmin_et(satir['userId'], satir['movieId'])
            gercek.append(satir['rating'])
            tahminler.append(tahmin)
        except Exception:
            continue
    
    # Metrikleri hesapla
    metrikler = tum_metrikleri_hesapla(gercek, tahminler)
    metrikleri_yazdir(metrikler)
    
    # ----- 6. Faktör Analizi -----
    print("\n🔍 6. Gizli faktörler analiz ediliyor...")
    model.faktor_analizi(n_top=3)
    
    # ----- 7. Örnek Öneriler -----
    print("\n🎬 7. Örnek kullanıcı için öneriler...")
    
    # Rastgele bir kullanıcı seç
    ornek_kullanici = egitim['userId'].iloc[0]
    oneriler = model.oner(ornek_kullanici, n=5)
    
    print(f"\nKullanıcı {ornek_kullanici} için öneriler:")
    for i, (film_id, tahmin) in enumerate(oneriler, 1):
        film = filmler[filmler['movieId'] == film_id]
        if len(film) > 0:
            film_adi = film['title'].values[0]
            print(f"   {i}. ⭐ {tahmin:.2f} | {film_adi}")
    
    # ----- 8. Eğitim Geçmişi -----
    print("\n📈 8. Eğitim süreci:")
    print("\n   Epoch | RMSE")
    print("   " + "-" * 20)
    for i, rmse in enumerate(model.egitim_gecmisi):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"   {i+1:5} | {rmse:.4f}")
    
    # ----- Sonuç -----
    print("\n" + "=" * 60)
    print("✅ SVD ÖRNEĞİ TAMAMLANDI!")
    print("=" * 60)
    
    print("""
    📚 Öğrendiklerimiz:
    
    1. SVD, matris ayrıştırma yöntemidir
    2. Kullanıcı ve filmleri gizli faktörlerle temsil eder
    3. Tahmin = μ + b_u + b_i + P_u · Q_i
    4. SGD ile iteratif olarak eğitilir
    
    🎯 SVD'nin Avantajları:
    - Büyük veri setlerinde verimli
    - Gizli kalıpları keşfeder
    - Seyreklik problemini azaltır
    
    ⚠️ SVD'nin Dezavantajları:
    - Cold Start problemi var
    - Gizli faktörler yorumlaması zor
    - Hiperparametre ayarı gerektirir
    """)


if __name__ == "__main__":
    main()
