# ============================================
# 👥 İşbirlikçi Filtreleme (Collaborative Filtering)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Kullanıcıların veya filmlerin benzerliklerine göre öneri yapar.
# 
# İşbirlikçi Filtreleme nedir?
# "Sana benzer kullanıcılar şu filmleri sevdi!"
# veya
# "Bu filmi sevenler, şu filmleri de sevdi!"
# 
# İki türü var:
# 1. Kullanıcı Tabanlı (User-Based):
#    - Benzer kullanıcıları bul
#    - Onların sevdiği filmleri öner
# 
# 2. Öğe Tabanlı (Item-Based):
#    - Benzer filmleri bul
#    - Kullanıcının sevdiğine benzer filmleri öner
# 
# Avantajları:
# ✅ İçerik bilgisi gerektirmez
# ✅ Sürpriz keşifler yapabilir
# 
# Dezavantajları:
# ❌ Cold Start (yeni kullanıcı/film) problemi
# ❌ Seyrek (sparse) veride zorlanır
#
# 💡 Gerçek hayat örneği:
# Amazon: "Bu ürünü alanlar şunları da aldı"
# Spotify: "Senin için mix" (benzer zevklere sahip kullanıcılardan)
# ============================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_model import BaseModel
from ..similarity import kosinus_benzerligi, pearson_korelasyonu


class CollaborativeFilteringModel(BaseModel):
    """
    🎯 Bu sınıf ne yapar?
    Kullanıcı veya öğe benzerliğine göre öneri yapar.
    
    📊 Türler:
    - "kullanici": Kullanıcı tabanlı (user-based)
    - "film": Öğe tabanlı (item-based)
    
    💡 Kullanım:
    >>> model = CollaborativeFilteringModel(tur="kullanici")
    >>> model.egit(filmler, puanlar)
    >>> oneriler = model.oner(kullanici_id=1, n=5)
    """
    
    def __init__(self, 
                 isim: str = "İşbirlikçi Filtreleme",
                 tur: str = "kullanici",
                 k_komsu: int = 20,
                 benzerlik_yontemi: str = "kosinus"):
        """
        🎯 Model nesnesini oluşturur.
        
        📥 Parametreler:
        - isim (str): Model ismi
        - tur (str): "kullanici" veya "film"
        - k_komsu (int): Kaç benzer komşu kullanılacak
        - benzerlik_yontemi (str): "kosinus" veya "pearson"
        """
        
        super().__init__(isim)
        
        # Model türü
        if tur not in ["kullanici", "film"]:
            raise ValueError("❌ Tür 'kullanici' veya 'film' olmalı!")
        self.tur = tur
        
        # Kaç komşu kullanılacak (K-NN için)
        self.k_komsu = k_komsu
        
        # Benzerlik yöntemi
        self.benzerlik_yontemi = benzerlik_yontemi
        
        # Kullanıcı-Film puan matrisi
        self.puan_matrisi = None
        
        # Benzerlik matrisi
        self.benzerlik_matrisi = None
        
        # ID ↔ İndeks eşleşmeleri
        self.kullanici_id_to_idx = {}
        self.idx_to_kullanici_id = {}
        self.film_id_to_idx = {}
        self.idx_to_film_id = {}
        
        print(f"   Tür: {self.tur.capitalize()} Tabanlı")
        print(f"   K Komşu: {self.k_komsu}")
    
    def egit(self, filmler: pd.DataFrame, puanlar: pd.DataFrame) -> None:
        """
        🎯 Bu metod ne yapar?
        Modeli verilen verilerle eğitir.
        
        Eğitim adımları:
        1. Kullanıcı-Film puan matrisini oluştur
        2. Benzerlik matrisini hesapla
        
        📥 Parametreler:
        - filmler (DataFrame): Film tablosu
        - puanlar (DataFrame): Puan tablosu
        """
        
        print(f"🔄 {self.isim} eğitimi başlıyor...")
        
        # Verileri kaydet
        self.filmler = filmler.copy()
        self.puanlar = puanlar.copy()
        
        # ----- Adım 1: Puan Matrisini Oluştur -----
        print("   📊 Puan matrisi oluşturuluyor...")
        self._puan_matrisi_olustur()
        
        # ----- Adım 2: Benzerlik Matrisini Hesapla -----
        print("   🔗 Benzerlik matrisi hesaplanıyor...")
        self._benzerlik_matrisi_hesapla()
        
        # Eğitim tamamlandı
        self.egitildi_mi = True
        
        # İstatistikleri kaydet
        self.parametreler = {
            'tur': self.tur,
            'k_komsu': self.k_komsu,
            'benzerlik_yontemi': self.benzerlik_yontemi,
            'matris_boyut': self.puan_matrisi.shape
        }
        
        print(f"✅ {self.isim} eğitimi tamamlandı!")
    
    def _puan_matrisi_olustur(self) -> None:
        """
        🎯 Bu metod ne yapar?
        Kullanıcı-Film puan matrisini oluşturur.
        
        Matris yapısı:
                    Film1  Film2  Film3  Film4
        Kullanıcı1   5.0    3.0    NaN    4.0
        Kullanıcı2   NaN    4.0    5.0    NaN
        Kullanıcı3   3.0    NaN    4.0    5.0
        
        NaN = Puanlanmamış
        """
        
        # ID ↔ İndeks eşleşmelerini oluştur
        kullanicilar = sorted(self.puanlar['userId'].unique())
        filmler_ids = sorted(self.puanlar['movieId'].unique())
        
        for idx, uid in enumerate(kullanicilar):
            self.kullanici_id_to_idx[uid] = idx
            self.idx_to_kullanici_id[idx] = uid
        
        for idx, mid in enumerate(filmler_ids):
            self.film_id_to_idx[mid] = idx
            self.idx_to_film_id[idx] = mid
        
        # Matrisi oluştur (NaN ile doldur)
        n_kullanici = len(kullanicilar)
        n_film = len(filmler_ids)
        self.puan_matrisi = np.full((n_kullanici, n_film), np.nan)
        
        # Puanları doldur
        for _, satir in self.puanlar.iterrows():
            k_idx = self.kullanici_id_to_idx[satir['userId']]
            f_idx = self.film_id_to_idx[satir['movieId']]
            self.puan_matrisi[k_idx, f_idx] = satir['rating']
        
        print(f"      Boyut: {n_kullanici} kullanıcı × {n_film} film")
    
    def _benzerlik_matrisi_hesapla(self) -> None:
        """
        🎯 Bu metod ne yapar?
        Tüm kullanıcılar veya filmler arası benzerliği hesaplar.
        """
        
        if self.tur == "kullanici":
            # Kullanıcılar arası benzerlik
            matris = self.puan_matrisi
        else:
            # Filmler arası benzerlik (matrisin transpozu)
            matris = self.puan_matrisi.T
        
        n = matris.shape[0]
        self.benzerlik_matrisi = np.zeros((n, n))
        
        # Her çift için benzerlik hesapla
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    self.benzerlik_matrisi[i, j] = 1.0
                else:
                    # Ortak puanlanmış öğeleri bul
                    maske = ~np.isnan(matris[i]) & ~np.isnan(matris[j])
                    
                    if maske.sum() < 2:
                        # Yeterli ortak öğe yok
                        benzerlik = 0.0
                    else:
                        v1 = matris[i][maske]
                        v2 = matris[j][maske]
                        
                        if self.benzerlik_yontemi == "kosinus":
                            benzerlik = kosinus_benzerligi(v1, v2)
                        else:
                            benzerlik = pearson_korelasyonu(v1, v2)
                    
                    # Simetrik matris
                    self.benzerlik_matrisi[i, j] = benzerlik
                    self.benzerlik_matrisi[j, i] = benzerlik
    
    def tahmin_et(self, kullanici_id: int, film_id: int) -> float:
        """
        🎯 Bu metod ne yapar?
        Kullanıcının filme vereceği puanı tahmin eder.
        
        Nasıl çalışır?
        1. En benzer K komşuyu bul
        2. Komşuların puanlarının ağırlıklı ortalamasını al
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - film_id (int): Film ID
        
        📤 Döndürdüğü:
        - tahmin (float): Tahmini puan
        """
        
        self.egitim_kontrolu()
        
        # ID'leri kontrol et
        if kullanici_id not in self.kullanici_id_to_idx:
            return self.puanlar['rating'].mean()
        
        if film_id not in self.film_id_to_idx:
            return self.puanlar['rating'].mean()
        
        k_idx = self.kullanici_id_to_idx[kullanici_id]
        f_idx = self.film_id_to_idx[film_id]
        
        if self.tur == "kullanici":
            # Kullanıcı tabanlı tahmin
            return self._kullanici_tabanli_tahmin(k_idx, f_idx)
        else:
            # Film tabanlı tahmin
            return self._film_tabanli_tahmin(k_idx, f_idx)
    
    def _kullanici_tabanli_tahmin(self, k_idx: int, f_idx: int) -> float:
        """
        🎯 Kullanıcı tabanlı tahmin yapar.
        
        Formül:
        r(u,i) = r̄(u) + Σ(sim(u,v) × (r(v,i) - r̄(v))) / Σ|sim(u,v)|
        
        Türkçesi:
        Tahmin = Kullanıcı ortalaması + Komşuların sapmaları ağırlıklı ort.
        """
        
        # Hedef kullanıcının benzerliklerini al
        benzerlikler = self.benzerlik_matrisi[k_idx]
        
        # Bu filmi puanlayan kullanıcıları bul
        film_puanlari = self.puan_matrisi[:, f_idx]
        puanlayan_maske = ~np.isnan(film_puanlari)
        
        if puanlayan_maske.sum() == 0:
            return self.puanlar['rating'].mean()
        
        # En benzer K komşuyu seç (kendisi hariç, filmi puanlayanlar)
        puanlayan_idxler = np.where(puanlayan_maske)[0]
        puanlayan_idxler = puanlayan_idxler[puanlayan_idxler != k_idx]
        
        if len(puanlayan_idxler) == 0:
            return self.puanlar['rating'].mean()
        
        # Benzerlik sırasına göre sırala
        sirali = sorted(
            [(idx, benzerlikler[idx]) for idx in puanlayan_idxler],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:self.k_komsu]
        
        # Ağırlıklı ortalama hesapla
        toplam_agirlik = 0
        agirlikli_toplam = 0
        
        # Hedef kullanıcının ortalaması
        kullanici_puanlari = self.puan_matrisi[k_idx]
        kullanici_ort = np.nanmean(kullanici_puanlari)
        
        for komsu_idx, benzerlik in sirali:
            if benzerlik <= 0:
                continue
            
            # Komşunun bu filme verdiği puan
            komsu_puan = film_puanlari[komsu_idx]
            
            # Komşunun ortalaması
            komsu_puanlari = self.puan_matrisi[komsu_idx]
            komsu_ort = np.nanmean(komsu_puanlari)
            
            # Sapmayı hesapla
            sapma = komsu_puan - komsu_ort
            
            agirlikli_toplam += benzerlik * sapma
            toplam_agirlik += abs(benzerlik)
        
        if toplam_agirlik == 0:
            return kullanici_ort
        
        # Tahmini hesapla
        tahmin = kullanici_ort + (agirlikli_toplam / toplam_agirlik)
        
        # Aralığı kontrol et
        tahmin = max(0.5, min(5.0, tahmin))
        
        return float(tahmin)
    
    def _film_tabanli_tahmin(self, k_idx: int, f_idx: int) -> float:
        """
        🎯 Film tabanlı tahmin yapar.
        
        Formül:
        r(u,i) = Σ(sim(i,j) × r(u,j)) / Σ|sim(i,j)|
        
        Türkçesi:
        Tahmin = Benzer filmlere verilen puanların ağırlıklı ortalaması
        """
        
        # Hedef filmin benzerliklerini al
        benzerlikler = self.benzerlik_matrisi[f_idx]
        
        # Kullanıcının puanladığı filmler
        kullanici_puanlari = self.puan_matrisi[k_idx]
        puanlanan_maske = ~np.isnan(kullanici_puanlari)
        
        if puanlanan_maske.sum() == 0:
            return self.puanlar['rating'].mean()
        
        # En benzer K filmi seç (kendisi hariç)
        puanlanan_idxler = np.where(puanlanan_maske)[0]
        puanlanan_idxler = puanlanan_idxler[puanlanan_idxler != f_idx]
        
        if len(puanlanan_idxler) == 0:
            return self.puanlar['rating'].mean()
        
        # Benzerlik sırasına göre sırala
        sirali = sorted(
            [(idx, benzerlikler[idx]) for idx in puanlanan_idxler],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:self.k_komsu]
        
        # Ağırlıklı ortalama hesapla
        toplam_agirlik = 0
        agirlikli_toplam = 0
        
        for film_idx, benzerlik in sirali:
            if benzerlik <= 0:
                continue
            
            puan = kullanici_puanlari[film_idx]
            agirlikli_toplam += benzerlik * puan
            toplam_agirlik += abs(benzerlik)
        
        if toplam_agirlik == 0:
            return np.nanmean(kullanici_puanlari)
        
        tahmin = agirlikli_toplam / toplam_agirlik
        tahmin = max(0.5, min(5.0, tahmin))
        
        return float(tahmin)
    
    def oner(self, kullanici_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Kullanıcıya n tane film önerir.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - n (int): Kaç film önerilecek
        
        📤 Döndürdüğü:
        - oneriler (list): [(film_id, tahmin), ...] listesi
        """
        
        self.egitim_kontrolu()
        
        # Kullanıcı kontrolü
        if kullanici_id not in self.kullanici_id_to_idx:
            print(f"⚠️ Kullanıcı {kullanici_id} bulunamadı!")
            return []
        
        k_idx = self.kullanici_id_to_idx[kullanici_id]
        
        # Kullanıcının puanladığı filmler
        puanlanan = set(np.where(~np.isnan(self.puan_matrisi[k_idx]))[0])
        
        # Tüm puanlanmamış filmler için tahmin yap
        tahminler = []
        
        for f_idx in range(self.puan_matrisi.shape[1]):
            if f_idx not in puanlanan:
                film_id = self.idx_to_film_id[f_idx]
                tahmin = self.tahmin_et(kullanici_id, film_id)
                tahminler.append((film_id, tahmin))
        
        # Puana göre sırala
        tahminler.sort(key=lambda x: x[1], reverse=True)
        
        return tahminler[:n]
    
    def benzer_kullanicilari_bul(self, kullanici_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Bir kullanıcıya en benzer kullanıcıları bulur.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - k (int): Kaç benzer kullanıcı
        
        📤 Döndürdüğü:
        - benzerler (list): [(kullanici_id, benzerlik), ...] listesi
        """
        
        self.egitim_kontrolu()
        
        if self.tur != "kullanici":
            print("⚠️ Bu fonksiyon sadece kullanıcı tabanlı modelde çalışır!")
            return []
        
        if kullanici_id not in self.kullanici_id_to_idx:
            return []
        
        k_idx = self.kullanici_id_to_idx[kullanici_id]
        benzerlikler = self.benzerlik_matrisi[k_idx]
        
        # Sırala (kendisi hariç)
        benzerler = []
        for idx, benz in enumerate(benzerlikler):
            if idx != k_idx:
                uid = self.idx_to_kullanici_id[idx]
                benzerler.append((uid, benz))
        
        benzerler.sort(key=lambda x: x[1], reverse=True)
        
        return benzerler[:k]
    
    def benzer_filmleri_bul(self, film_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Bir filme en benzer filmleri bulur.
        
        📥 Parametreler:
        - film_id (int): Film ID
        - k (int): Kaç benzer film
        
        📤 Döndürdüğü:
        - benzerler (list): [(film_id, benzerlik), ...] listesi
        """
        
        self.egitim_kontrolu()
        
        if self.tur != "film":
            print("⚠️ Bu fonksiyon sadece film tabanlı modelde çalışır!")
            return []
        
        if film_id not in self.film_id_to_idx:
            return []
        
        f_idx = self.film_id_to_idx[film_id]
        benzerlikler = self.benzerlik_matrisi[f_idx]
        
        # Sırala (kendisi hariç)
        benzerler = []
        for idx, benz in enumerate(benzerlikler):
            if idx != f_idx:
                mid = self.idx_to_film_id[idx]
                benzerler.append((mid, benz))
        
        benzerler.sort(key=lambda x: x[1], reverse=True)
        
        return benzerler[:k]


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 İşbirlikçi Filtreleme testi\n")
    
    # Test verisi oluştur
    puanlar = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        'movieId': [1, 2, 3, 1, 2, 4, 2, 3, 4, 1, 3],
        'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.0, 5.0, 4.0]
    })
    
    filmler = pd.DataFrame({
        'movieId': [1, 2, 3, 4],
        'title': ['Film A', 'Film B', 'Film C', 'Film D'],
        'genres': ['Action', 'Comedy', 'Drama', 'Action|Comedy']
    })
    
    # Kullanıcı tabanlı model
    print("=" * 50)
    model_kullanici = CollaborativeFilteringModel(tur="kullanici")
    model_kullanici.egit(filmler, puanlar)
    
    print("\n👥 En benzer kullanıcılar (Kullanıcı 1 için):")
    benzerler = model_kullanici.benzer_kullanicilari_bul(1, k=3)
    for uid, benz in benzerler:
        print(f"   Kullanıcı {uid}: {benz:.3f}")
    
    print("\n🎬 Öneriler (Kullanıcı 1 için):")
    oneriler = model_kullanici.oner(1, n=2)
    for film_id, tahmin in oneriler:
        print(f"   Film {film_id}: {tahmin:.2f}")
    
    # Film tabanlı model
    print("\n" + "=" * 50)
    model_film = CollaborativeFilteringModel(tur="film")
    model_film.egit(filmler, puanlar)
    
    print("\n🎬 En benzer filmler (Film 1 için):")
    benzer_filmler = model_film.benzer_filmleri_bul(1, k=3)
    for mid, benz in benzer_filmler:
        print(f"   Film {mid}: {benz:.3f}")
    
    print("\n✅ İşbirlikçi Filtreleme testi tamamlandı!")
