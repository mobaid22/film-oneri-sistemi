# ============================================
# 🧪 Benzerlik Testleri
# ============================================
# 
# Bu dosya src/similarity.py modülündeki
# fonksiyonları test eder.
#
# Testleri çalıştırmak için:
# pytest tests/test_similarity.py -v
# ============================================

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.similarity import (
    kosinus_benzerligi,
    pearson_korelasyonu,
    oklid_uzakligi,
    oklid_benzerligi,
    manhattan_uzakligi,
    jaccard_benzerligi
)


class TestKosinusBenzerligi:
    """Kosinüs benzerliği testleri"""
    
    def test_ayni_vektor(self):
        """
        🧪 Test: Aynı vektörün kendisiyle benzerliği 1 olmalı
        """
        v = [1, 2, 3, 4, 5]
        
        sonuc = kosinus_benzerligi(v, v)
        
        assert abs(sonuc - 1.0) < 0.0001
    
    def test_zit_vektor(self):
        """
        🧪 Test: Zıt vektörlerin benzerliği -1 olmalı
        """
        v1 = [1, 0]
        v2 = [-1, 0]
        
        sonuc = kosinus_benzerligi(v1, v2)
        
        assert abs(sonuc - (-1.0)) < 0.0001
    
    def test_dik_vektor(self):
        """
        🧪 Test: Dik vektörlerin benzerliği 0 olmalı
        """
        v1 = [1, 0]
        v2 = [0, 1]
        
        sonuc = kosinus_benzerligi(v1, v2)
        
        assert abs(sonuc) < 0.0001
    
    def test_sifir_vektor(self):
        """
        🧪 Test: Sıfır vektörü için 0 dönmeli
        """
        v1 = [0, 0, 0]
        v2 = [1, 2, 3]
        
        sonuc = kosinus_benzerligi(v1, v2)
        
        assert sonuc == 0.0
    
    def test_boyut_hatasi(self):
        """
        🧪 Test: Farklı boyutlarda hata vermeli
        """
        v1 = [1, 2, 3]
        v2 = [1, 2]
        
        with pytest.raises(ValueError):
            kosinus_benzerligi(v1, v2)
    
    def test_bilinen_deger(self):
        """
        🧪 Test: Bilinen bir değerle kontrol
        
        v1 = [3, 4]  ||v1|| = 5
        v2 = [4, 3]  ||v2|| = 5
        v1·v2 = 12 + 12 = 24
        cos = 24 / 25 = 0.96
        """
        v1 = [3, 4]
        v2 = [4, 3]
        
        sonuc = kosinus_benzerligi(v1, v2)
        
        assert abs(sonuc - 0.96) < 0.01


class TestPearsonKorelasyonu:
    """Pearson korelasyonu testleri"""
    
    def test_mükemmel_korelasyon(self):
        """
        🧪 Test: Aynı eğilim → korelasyon = 1
        """
        v1 = [1, 2, 3, 4, 5]
        v2 = [2, 4, 6, 8, 10]  # v1'in 2 katı
        
        sonuc = pearson_korelasyonu(v1, v2)
        
        assert abs(sonuc - 1.0) < 0.0001
    
    def test_negatif_korelasyon(self):
        """
        🧪 Test: Ters eğilim → korelasyon = -1
        """
        v1 = [1, 2, 3, 4, 5]
        v2 = [5, 4, 3, 2, 1]  # v1'in tersi
        
        sonuc = pearson_korelasyonu(v1, v2)
        
        assert abs(sonuc - (-1.0)) < 0.0001
    
    def test_farkli_ortalama(self):
        """
        🧪 Test: Farklı ortalamalarla aynı eğilim
        
        Pearson ortalamaları normalize eder,
        bu yüzden farklı ortalamalar korelasyonu etkilemez.
        """
        v1 = [1, 2, 3]  # ortalama = 2
        v2 = [11, 12, 13]  # ortalama = 12
        
        sonuc = pearson_korelasyonu(v1, v2)
        
        assert abs(sonuc - 1.0) < 0.0001
    
    def test_sabit_vektor(self):
        """
        🧪 Test: Sabit vektörde (std=0) 0 dönmeli
        """
        v1 = [5, 5, 5, 5]
        v2 = [1, 2, 3, 4]
        
        sonuc = pearson_korelasyonu(v1, v2)
        
        assert sonuc == 0.0


class TestOklidUzakligi:
    """Öklid uzaklığı testleri"""
    
    def test_ayni_nokta(self):
        """
        🧪 Test: Aynı noktanın kendisine uzaklığı 0
        """
        v = [1, 2, 3]
        
        sonuc = oklid_uzakligi(v, v)
        
        assert sonuc == 0.0
    
    def test_bilinen_deger(self):
        """
        🧪 Test: 3-4-5 üçgeni
        
        (0,0) ile (3,4) arası uzaklık = 5
        """
        v1 = [0, 0]
        v2 = [3, 4]
        
        sonuc = oklid_uzakligi(v1, v2)
        
        assert sonuc == 5.0
    
    def test_oklid_benzerligi(self):
        """
        🧪 Test: Benzerlik = 1 / (1 + uzaklık)
        
        uzaklık = 0 → benzerlik = 1
        """
        v = [1, 2, 3]
        
        sonuc = oklid_benzerligi(v, v)
        
        assert sonuc == 1.0


class TestManhattanUzakligi:
    """Manhattan uzaklığı testleri"""
    
    def test_ayni_nokta(self):
        """
        🧪 Test: Aynı noktaya uzaklık 0
        """
        v = [1, 2, 3]
        
        sonuc = manhattan_uzakligi(v, v)
        
        assert sonuc == 0.0
    
    def test_bilinen_deger(self):
        """
        🧪 Test: Manhattan = |3-0| + |4-0| = 7
        """
        v1 = [0, 0]
        v2 = [3, 4]
        
        sonuc = manhattan_uzakligi(v1, v2)
        
        assert sonuc == 7.0


class TestJaccardBenzerligi:
    """Jaccard benzerliği testleri"""
    
    def test_ayni_kume(self):
        """
        🧪 Test: Aynı kümelerin benzerliği 1
        """
        k = {1, 2, 3}
        
        sonuc = jaccard_benzerligi(k, k)
        
        assert sonuc == 1.0
    
    def test_ayrik_kumeler(self):
        """
        🧪 Test: Kesişimi olmayan kümelerin benzerliği 0
        """
        k1 = {1, 2, 3}
        k2 = {4, 5, 6}
        
        sonuc = jaccard_benzerligi(k1, k2)
        
        assert sonuc == 0.0
    
    def test_bilinen_deger(self):
        """
        🧪 Test: Kesişim = {3,4,5}, Birleşim = {1,2,3,4,5,6,7}
        Jaccard = 3/7 ≈ 0.4286
        """
        k1 = {1, 2, 3, 4, 5}
        k2 = {3, 4, 5, 6, 7}
        
        sonuc = jaccard_benzerligi(k1, k2)
        
        assert abs(sonuc - 3/7) < 0.001
    
    def test_bos_kumeler(self):
        """
        🧪 Test: Boş kümelerin benzerliği 0
        """
        k1 = set()
        k2 = set()
        
        sonuc = jaccard_benzerligi(k1, k2)
        
        assert sonuc == 0.0


# ============================================
# 🚀 Testleri Çalıştır
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
