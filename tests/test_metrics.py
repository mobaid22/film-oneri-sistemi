# ============================================
# 🧪 Metrik Testleri
# ============================================
# 
# Bu dosya src/metrics.py modülündeki
# fonksiyonları test eder.
#
# Testleri çalıştırmak için:
# pytest tests/test_metrics.py -v
# ============================================

import pytest
import numpy as np
import sys
import os

# src klasörünü import path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import (
    rmse_hesapla,
    mae_hesapla,
    mse_hesapla,
    r2_hesapla,
    mape_hesapla,
    precision_hesapla,
    recall_hesapla,
    f1_hesapla,
    tum_metrikleri_hesapla
)


class TestRMSE:
    """RMSE fonksiyonu için testler"""
    
    def test_rmse_mükemmel_tahmin(self):
        """
        🧪 Test: Mükemmel tahmin durumunda RMSE = 0 olmalı
        """
        gercek = [1, 2, 3, 4, 5]
        tahmin = [1, 2, 3, 4, 5]
        
        sonuc = rmse_hesapla(gercek, tahmin)
        
        assert sonuc == 0.0, "Mükemmel tahmin için RMSE 0 olmalı"
    
    def test_rmse_bilinen_deger(self):
        """
        🧪 Test: Bilinen değerlerle RMSE hesaplama
        
        gercek = [3, 4]
        tahmin = [2, 5]
        hatalar = [1, -1]
        kare_hatalar = [1, 1]
        ortalama = 1
        rmse = √1 = 1
        """
        gercek = [3, 4]
        tahmin = [2, 5]
        
        sonuc = rmse_hesapla(gercek, tahmin)
        
        assert sonuc == 1.0, f"RMSE = 1 olmalı, ama {sonuc} bulundu"
    
    def test_rmse_pozitif(self):
        """
        🧪 Test: RMSE her zaman pozitif olmalı
        """
        gercek = [5, 4, 3, 2, 1]
        tahmin = [1, 2, 3, 4, 5]
        
        sonuc = rmse_hesapla(gercek, tahmin)
        
        assert sonuc > 0, "RMSE pozitif olmalı"
    
    def test_rmse_boyut_uyusmazligi(self):
        """
        🧪 Test: Farklı boyutlarda hata vermeli
        """
        gercek = [1, 2, 3]
        tahmin = [1, 2]
        
        with pytest.raises(ValueError):
            rmse_hesapla(gercek, tahmin)


class TestMAE:
    """MAE fonksiyonu için testler"""
    
    def test_mae_mükemmel_tahmin(self):
        """
        🧪 Test: Mükemmel tahmin durumunda MAE = 0 olmalı
        """
        gercek = [1, 2, 3]
        tahmin = [1, 2, 3]
        
        sonuc = mae_hesapla(gercek, tahmin)
        
        assert sonuc == 0.0
    
    def test_mae_bilinen_deger(self):
        """
        🧪 Test: MAE = ortalama mutlak hata
        
        gercek = [4, 5]
        tahmin = [3, 6]
        hatalar = [1, -1]
        mutlak = [1, 1]
        mae = 1
        """
        gercek = [4, 5]
        tahmin = [3, 6]
        
        sonuc = mae_hesapla(gercek, tahmin)
        
        assert sonuc == 1.0


class TestMSE:
    """MSE fonksiyonu için testler"""
    
    def test_mse_mükemmel_tahmin(self):
        """
        🧪 Test: Mükemmel tahmin için MSE = 0
        """
        gercek = [1, 2, 3]
        tahmin = [1, 2, 3]
        
        sonuc = mse_hesapla(gercek, tahmin)
        
        assert sonuc == 0.0
    
    def test_mse_rmse_iliskisi(self):
        """
        🧪 Test: MSE = RMSE²
        """
        gercek = [3, 4, 5]
        tahmin = [2, 4, 6]
        
        mse = mse_hesapla(gercek, tahmin)
        rmse = rmse_hesapla(gercek, tahmin)
        
        assert abs(mse - rmse ** 2) < 0.0001, "MSE = RMSE² olmalı"


class TestR2:
    """R² fonksiyonu için testler"""
    
    def test_r2_mükemmel_tahmin(self):
        """
        🧪 Test: Mükemmel tahmin için R² = 1
        """
        gercek = [1, 2, 3, 4, 5]
        tahmin = [1, 2, 3, 4, 5]
        
        sonuc = r2_hesapla(gercek, tahmin)
        
        assert sonuc == 1.0
    
    def test_r2_aralik(self):
        """
        🧪 Test: R² genellikle 0-1 arasında olmalı
        (kötü modeller için negatif olabilir)
        """
        gercek = [3, 4, 5, 6, 7]
        tahmin = [3.1, 3.9, 5.2, 5.8, 7.1]
        
        sonuc = r2_hesapla(gercek, tahmin)
        
        assert 0 <= sonuc <= 1, f"R² 0-1 arasında olmalı, ama {sonuc} bulundu"


class TestMAPE:
    """MAPE fonksiyonu için testler"""
    
    def test_mape_mükemmel_tahmin(self):
        """
        🧪 Test: Mükemmel tahmin için MAPE = 0
        """
        gercek = [100, 200, 300]
        tahmin = [100, 200, 300]
        
        sonuc = mape_hesapla(gercek, tahmin)
        
        assert sonuc == 0.0
    
    def test_mape_bilinen_deger(self):
        """
        🧪 Test: %10 hata durumu
        
        gercek = [100]
        tahmin = [90]
        hata = 10
        yuzde = 10%
        """
        gercek = [100]
        tahmin = [90]
        
        sonuc = mape_hesapla(gercek, tahmin)
        
        assert sonuc == 10.0
    
    def test_mape_sifir_hatasi(self):
        """
        🧪 Test: Gerçek değer 0 içerirse hata vermeli
        """
        gercek = [100, 0, 200]
        tahmin = [100, 50, 200]
        
        with pytest.raises(ValueError):
            mape_hesapla(gercek, tahmin)


class TestPrecisionRecall:
    """Precision ve Recall testleri"""
    
    def test_precision_bilinen_deger(self):
        """
        🧪 Test: 5 öneriden 3'ü doğru → Precision = 0.6
        """
        onerilen = {1, 2, 3, 4, 5}
        begenilenler = {1, 2, 3, 8, 9}
        
        sonuc = precision_hesapla(onerilen, begenilenler)
        
        assert sonuc == 0.6
    
    def test_recall_bilinen_deger(self):
        """
        🧪 Test: 5 beğenilenin 3'ü önerildi → Recall = 0.6
        """
        onerilen = {1, 2, 3, 4, 5}
        begenilenler = {1, 2, 3, 8, 9}
        
        sonuc = recall_hesapla(onerilen, begenilenler)
        
        assert sonuc == 0.6
    
    def test_f1_hesaplama(self):
        """
        🧪 Test: F1 = 2 * P * R / (P + R)
        """
        precision = 0.8
        recall = 0.6
        
        beklenen = 2 * 0.8 * 0.6 / (0.8 + 0.6)  # 0.6857...
        sonuc = f1_hesapla(precision, recall)
        
        assert abs(sonuc - beklenen) < 0.001
    
    def test_f1_dengeli(self):
        """
        🧪 Test: P = R ise F1 de aynı olmalı
        """
        precision = 0.7
        recall = 0.7
        
        sonuc = f1_hesapla(precision, recall)
        
        assert sonuc == 0.7


class TestTumMetrikler:
    """tum_metrikleri_hesapla fonksiyonu testi"""
    
    def test_tum_metrikler_anahtarlar(self):
        """
        🧪 Test: Tüm beklenen metrikler döndürülmeli
        """
        gercek = [3, 4, 5]
        tahmin = [3.1, 3.9, 5.1]
        
        sonuc = tum_metrikleri_hesapla(gercek, tahmin)
        
        beklenen_anahtarlar = ['RMSE', 'MAE', 'MSE', 'R2']
        
        for anahtar in beklenen_anahtarlar:
            assert anahtar in sonuc, f"{anahtar} sonuçlarda olmalı"


# ============================================
# 🚀 Testleri Çalıştır
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
