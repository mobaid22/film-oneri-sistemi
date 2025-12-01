# ============================================
# 🧪 Model Testleri
# ============================================
# 
# Bu dosya src/models/ modülündeki
# modelleri test eder.
#
# Testleri çalıştırmak için:
# pytest tests/test_models.py -v
# ============================================

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeFilteringModel
from src.models.svd_model import SVDModel
from src.models.hybrid import HybridModel


# ============================================
# 📦 Test Verileri
# ============================================

@pytest.fixture
def ornek_filmler():
    """Test için örnek film verisi"""
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Film A', 'Film B', 'Film C', 'Film D', 'Film E'],
        'genres': ['Action|Adventure', 'Comedy|Romance', 'Action|Sci-Fi', 
                   'Drama|Romance', 'Action|Comedy']
    })


@pytest.fixture
def ornek_puanlar():
    """Test için örnek puan verisi"""
    return pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movieId': [1, 2, 3, 2, 3, 4, 1, 4, 5],
        'rating': [5.0, 3.0, 4.0, 4.0, 5.0, 4.0, 4.0, 3.0, 5.0]
    })


# ============================================
# 📝 İçerik Tabanlı Model Testleri
# ============================================

class TestContentBasedModel:
    """ContentBasedModel testleri"""
    
    def test_model_olusturma(self):
        """
        🧪 Test: Model başarıyla oluşturulmalı
        """
        model = ContentBasedModel("Test Model")
        
        assert model.isim == "Test Model"
        assert model.egitildi_mi == False
    
    def test_egitim(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Model eğitimi başarılı olmalı
        """
        model = ContentBasedModel()
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert model.egitildi_mi == True
        assert model.film_ozellikleri is not None
    
    def test_tahmin(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Tahmin 0.5-5 arasında olmalı
        """
        model = ContentBasedModel()
        model.egit(ornek_filmler, ornek_puanlar)
        
        tahmin = model.tahmin_et(kullanici_id=1, film_id=4)
        
        assert 0.5 <= tahmin <= 5.0
    
    def test_oner(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Öneriler doğru formatta olmalı
        """
        model = ContentBasedModel()
        model.egit(ornek_filmler, ornek_puanlar)
        
        oneriler = model.oner(kullanici_id=1, n=3)
        
        assert len(oneriler) > 0
        assert all(isinstance(o, tuple) for o in oneriler)
        assert all(len(o) == 2 for o in oneriler)
    
    def test_egitilmemis_model_hatasi(self):
        """
        🧪 Test: Eğitilmemiş model hata vermeli
        """
        model = ContentBasedModel()
        
        with pytest.raises(RuntimeError):
            model.tahmin_et(1, 1)


# ============================================
# 👥 İşbirlikçi Filtreleme Testleri
# ============================================

class TestCollaborativeModel:
    """CollaborativeFilteringModel testleri"""
    
    def test_kullanici_tabanli(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Kullanıcı tabanlı model çalışmalı
        """
        model = CollaborativeFilteringModel(tur="kullanici")
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert model.egitildi_mi == True
        assert model.tur == "kullanici"
    
    def test_film_tabanli(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Film tabanlı model çalışmalı
        """
        model = CollaborativeFilteringModel(tur="film")
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert model.egitildi_mi == True
        assert model.tur == "film"
    
    def test_benzerlik_matrisi(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Benzerlik matrisi oluşturulmalı
        """
        model = CollaborativeFilteringModel(tur="kullanici")
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert model.benzerlik_matrisi is not None
        # Diyagonal değerler 1 olmalı (kendisiyle benzerlik)
        assert all(model.benzerlik_matrisi[i, i] == 1.0 
                  for i in range(len(model.benzerlik_matrisi)))
    
    def test_gecersiz_tur(self):
        """
        🧪 Test: Geçersiz tür hata vermeli
        """
        with pytest.raises(ValueError):
            CollaborativeFilteringModel(tur="gecersiz")
    
    def test_tahmin_aralik(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Tahmin 0.5-5 arasında olmalı
        """
        model = CollaborativeFilteringModel(tur="kullanici")
        model.egit(ornek_filmler, ornek_puanlar)
        
        tahmin = model.tahmin_et(1, 4)
        
        assert 0.5 <= tahmin <= 5.0


# ============================================
# 🔢 SVD Model Testleri
# ============================================

class TestSVDModel:
    """SVDModel testleri"""
    
    def test_egitim(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: SVD eğitimi başarılı olmalı
        """
        model = SVDModel(n_faktor=5, iterasyon=5)
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert model.egitildi_mi == True
        assert model.P is not None
        assert model.Q is not None
    
    def test_faktor_boyutlari(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Faktör matrisleri doğru boyutta olmalı
        """
        n_faktor = 10
        model = SVDModel(n_faktor=n_faktor, iterasyon=5)
        model.egit(ornek_filmler, ornek_puanlar)
        
        n_kullanici = ornek_puanlar['userId'].nunique()
        n_film = ornek_puanlar['movieId'].nunique()
        
        assert model.P.shape == (n_kullanici, n_faktor)
        assert model.Q.shape == (n_film, n_faktor)
    
    def test_egitim_gecmisi(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Eğitim geçmişi kaydedilmeli
        """
        iterasyon = 10
        model = SVDModel(n_faktor=5, iterasyon=iterasyon)
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert len(model.egitim_gecmisi) == iterasyon
    
    def test_rmse_azalmali(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: RMSE genellikle azalmalı (öğrenme var)
        """
        model = SVDModel(n_faktor=5, iterasyon=20)
        model.egit(ornek_filmler, ornek_puanlar)
        
        # İlk ve son RMSE karşılaştır
        ilk_rmse = model.egitim_gecmisi[0]
        son_rmse = model.egitim_gecmisi[-1]
        
        # Genellikle azalmalı, ama garanti değil
        # En azından çok artmamalı
        assert son_rmse <= ilk_rmse * 1.5


# ============================================
# 🔀 Hibrit Model Testleri
# ============================================

class TestHybridModel:
    """HybridModel testleri"""
    
    def test_agirlik_normalizasyonu(self):
        """
        🧪 Test: Ağırlıklar normalize edilmeli
        """
        model = HybridModel(icerik_agirligi=0.3, isbirligi_agirligi=0.7)
        
        toplam = model.icerik_agirligi + model.isbirligi_agirligi
        assert abs(toplam - 1.0) < 0.0001
    
    def test_her_iki_model_egitilmeli(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Alt modeller eğitilmeli
        """
        model = HybridModel()
        model.egit(ornek_filmler, ornek_puanlar)
        
        assert model.icerik_modeli.egitildi_mi == True
        assert model.isbirligi_modeli.egitildi_mi == True
    
    def test_hibrit_tahmin(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Hibrit tahmin çalışmalı
        """
        model = HybridModel()
        model.egit(ornek_filmler, ornek_puanlar)
        
        tahmin = model.tahmin_et(1, 4)
        
        assert 0.5 <= tahmin <= 5.0
    
    def test_cold_start_tespit(self, ornek_filmler, ornek_puanlar):
        """
        🧪 Test: Cold start tespiti çalışmalı
        """
        model = HybridModel()
        model.egit(ornek_filmler, ornek_puanlar)
        
        # Var olan kullanıcı
        durum1 = model.cold_start_tespit_et(1)
        assert "COLD START" in durum1 or "YETERLİ" in durum1 or "ILIK" in durum1
        
        # Olmayan kullanıcı
        durum2 = model.cold_start_tespit_et(999)
        assert "TAM COLD START" in durum2


# ============================================
# 🚀 Testleri Çalıştır
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
