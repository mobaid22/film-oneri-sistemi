# ============================================
# 📝 İçerik Tabanlı Filtreleme (Content-Based)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Filmlerin içerik özelliklerine göre öneri yapar.
# 
# İçerik Tabanlı Filtreleme nedir?
# "Aksiyon filmi seviyorsan, başka aksiyon filmleri önerelim!"
# 
# Nasıl çalışır?
# 1. Filmlerin özelliklerini çıkar (tür, yönetmen, oyuncular)
# 2. Kullanıcının sevdiği filmlerin özelliklerini bul
# 3. Benzer özelliklere sahip filmleri öner
# 
# Avantajları:
# ✅ Yeni filmler için de çalışır (Cold Start çözer)
# ✅ Şeffaf: Neden önerdiğini açıklayabilir
# 
# Dezavantajları:
# ❌ Sadece benzer şeyler önerir (sürpriz yok)
# ❌ İçerik özellikleri gerektirir
#
# 💡 Gerçek hayat örneği:
# Netflix'te "Stranger Things" izledin.
# Sistem: "Sci-Fi + 80'ler + Gençler" özelliklerini buldu.
# Öneri: "E.T.", "Super 8", "Stand By Me"
# ============================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from .base_model import BaseModel
from ..similarity import kosinus_benzerligi


class ContentBasedModel(BaseModel):
    """
    🎯 Bu sınıf ne yapar?
    Film içeriklerine göre öneri yapar.
    
    Kullanıcının beğendiği filmlere benzer özelliklere
    sahip filmleri önerir.
    
    📊 Özellikler:
    - film_ozellikleri: Her filmin özellik vektörü
    - kullanici_profilleri: Her kullanıcının tercih vektörü
    
    💡 Kullanım:
    >>> model = ContentBasedModel()
    >>> model.egit(filmler, puanlar)
    >>> oneriler = model.oner(kullanici_id=1, n=5)
    """
    
    def __init__(self, isim: str = "İçerik Tabanlı Model"):
        """
        🎯 Model nesnesini oluşturur.
        
        📥 Parametreler:
        - isim (str): Model ismi
        """
        
        # Üst sınıfın (BaseModel) yapıcısını çağır
        super().__init__(isim)
        
        # Film özellik matrisi
        # Her satır bir film, her sütun bir özellik (tür)
        self.film_ozellikleri = None
        
        # Film ID → İndeks eşleşmesi
        self.film_id_to_idx = {}
        self.idx_to_film_id = {}
        
        # Tür listesi
        self.turler = []
        
        # Kullanıcı profilleri
        # Her kullanıcının hangi türleri sevdiğini gösterir
        self.kullanici_profilleri = {}
    
    def egit(self, filmler: pd.DataFrame, puanlar: pd.DataFrame) -> None:
        """
        🎯 Bu metod ne yapar?
        Modeli film ve puan verileriyle eğitir.
        
        Eğitim adımları:
        1. Film özellik matrisini oluştur (türlere göre)
        2. Her kullanıcının tercih profilini hesapla
        
        📥 Parametreler:
        - filmler (DataFrame): Film tablosu (movieId, title, genres)
        - puanlar (DataFrame): Puan tablosu (userId, movieId, rating)
        """
        
        print(f"🔄 {self.isim} eğitimi başlıyor...")
        
        # Verileri kaydet
        self.filmler = filmler.copy()
        self.puanlar = puanlar.copy()
        
        # ----- Adım 1: Film Özellik Matrisini Oluştur -----
        print("   📊 Film özellikleri çıkarılıyor...")
        self._film_ozelliklerini_olustur()
        
        # ----- Adım 2: Kullanıcı Profillerini Hesapla -----
        print("   👤 Kullanıcı profilleri oluşturuluyor...")
        self._kullanici_profillerini_olustur()
        
        # Eğitim tamamlandı
        self.egitildi_mi = True
        
        # İstatistikleri kaydet
        self.parametreler = {
            'film_sayisi': len(self.filmler),
            'tur_sayisi': len(self.turler),
            'kullanici_sayisi': len(self.kullanici_profilleri)
        }
        
        print(f"✅ {self.isim} eğitimi tamamlandı!")
        print(f"   🎬 {len(self.filmler):,} film, {len(self.turler)} tür")
        print(f"   👤 {len(self.kullanici_profilleri):,} kullanıcı profili")
    
    def _film_ozelliklerini_olustur(self) -> None:
        """
        🎯 Bu metod ne yapar?
        Her film için özellik vektörü oluşturur.
        
        Örnek:
        Film: "Toy Story" - Türler: Animation|Children|Comedy
        Özellik vektörü: [1, 0, 1, 1, 0, 0, ...] (her tür için 0 veya 1)
        
        Bu işleme "One-Hot Encoding" denir.
        """
        
        # Tüm benzersiz türleri bul
        tum_turler = set()
        for turler_str in self.filmler['genres']:
            if turler_str != "(no genres listed)":
                turler = turler_str.split('|')
                tum_turler.update(turler)
        
        # Türleri sırala (tutarlılık için)
        self.turler = sorted(list(tum_turler))
        
        # Film ID ↔ İndeks eşleşmelerini oluştur
        for idx, row in self.filmler.iterrows():
            film_id = row['movieId']
            self.film_id_to_idx[film_id] = idx
            self.idx_to_film_id[idx] = film_id
        
        # Özellik matrisi oluştur
        # Boyut: (film sayısı) x (tür sayısı)
        n_film = len(self.filmler)
        n_tur = len(self.turler)
        self.film_ozellikleri = np.zeros((n_film, n_tur))
        
        # Her film için özellikleri doldur
        for idx, row in self.filmler.iterrows():
            turler_str = row['genres']
            
            if turler_str != "(no genres listed)":
                film_turleri = turler_str.split('|')
                
                for tur in film_turleri:
                    if tur in self.turler:
                        # Bu türün sütun indeksini bul
                        tur_idx = self.turler.index(tur)
                        # 1 olarak işaretle
                        self.film_ozellikleri[idx, tur_idx] = 1
    
    def _kullanici_profillerini_olustur(self) -> None:
        """
        🎯 Bu metod ne yapar?
        Her kullanıcı için tercih profili oluşturur.
        
        Profil nedir?
        Kullanıcının hangi türleri ne kadar sevdiğini gösteren vektör.
        
        Nasıl hesaplanır?
        Kullanıcının yüksek puan verdiği filmlerin
        türlerini ağırlıklı ortalamasını alırız.
        
        Örnek:
        Kullanıcı Action filmlerine hep 5 vermiş → Action ağırlığı yüksek
        Kullanıcı Romance filmlerine hep 2 vermiş → Romance ağırlığı düşük
        """
        
        # Her kullanıcı için profil hesapla
        for kullanici_id in self.puanlar['userId'].unique():
            # Bu kullanıcının puanlarını al
            kullanici_puanlari = self.puanlar[
                self.puanlar['userId'] == kullanici_id
            ]
            
            # Profil vektörünü başlat (tüm türler için 0)
            profil = np.zeros(len(self.turler))
            toplam_puan = 0
            
            # Her puanlanan film için
            for _, satir in kullanici_puanlari.iterrows():
                film_id = satir['movieId']
                puan = satir['rating']
                
                # Film indeksini bul
                if film_id in self.film_id_to_idx:
                    film_idx = self.film_id_to_idx[film_id]
                    
                    # Filmin özelliklerini puanla çarp ve ekle
                    # Yüksek puanlı filmler daha çok etki eder
                    profil += self.film_ozellikleri[film_idx] * puan
                    toplam_puan += puan
            
            # Normalize et (ortalamasını al)
            if toplam_puan > 0:
                profil = profil / toplam_puan
            
            # Profili kaydet
            self.kullanici_profilleri[kullanici_id] = profil
    
    def tahmin_et(self, kullanici_id: int, film_id: int) -> float:
        """
        🎯 Bu metod ne yapar?
        Kullanıcının filme vereceği puanı tahmin eder.
        
        Nasıl çalışır?
        Kullanıcı profili ile film özellikleri arasındaki
        kosinüs benzerliğini hesaplar.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - film_id (int): Film ID
        
        📤 Döndürdüğü:
        - tahmin (float): 0.5 - 5.0 arası puan tahmini
        """
        
        # Eğitim kontrolü
        self.egitim_kontrolu()
        
        # Kullanıcı profili var mı?
        if kullanici_id not in self.kullanici_profilleri:
            # Yeni kullanıcı - ortalama puan döndür
            return self.puanlar['rating'].mean()
        
        # Film var mı?
        if film_id not in self.film_id_to_idx:
            return self.puanlar['rating'].mean()
        
        # Kullanıcı profilini al
        kullanici_profili = self.kullanici_profilleri[kullanici_id]
        
        # Film özelliklerini al
        film_idx = self.film_id_to_idx[film_id]
        film_ozellikleri = self.film_ozellikleri[film_idx]
        
        # Kosinüs benzerliği hesapla
        benzerlik = kosinus_benzerligi(kullanici_profili, film_ozellikleri)
        
        # Benzerliği 0.5-5.0 aralığına dönüştür
        # benzerlik: -1 ile 1 arası → puan: 0.5 ile 5 arası
        tahmin = 0.5 + (benzerlik + 1) * 2.25
        
        # Puan aralığını kontrol et
        tahmin = max(0.5, min(5.0, tahmin))
        
        return float(tahmin)
    
    def oner(self, kullanici_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Kullanıcıya n tane film önerir.
        
        Nasıl çalışır?
        1. Kullanıcının izlemediği filmleri bul
        2. Her film için puan tahmini yap
        3. En yüksek puanlı n filmi döndür
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - n (int): Kaç film önerilecek
        
        📤 Döndürdüğü:
        - oneriler (list): [(film_id, tahmin), ...] listesi
        """
        
        # Eğitim kontrolü
        self.egitim_kontrolu()
        
        # Kullanıcının zaten izlediği filmler
        izlenen = set(
            self.puanlar[self.puanlar['userId'] == kullanici_id]['movieId']
        )
        
        # Tüm filmler için tahmin yap
        tahminler = []
        
        for film_id in self.filmler['movieId']:
            # İzlenmemiş filmler için tahmin yap
            if film_id not in izlenen:
                tahmin = self.tahmin_et(kullanici_id, film_id)
                tahminler.append((film_id, tahmin))
        
        # Puana göre sırala (yüksekten düşüğe)
        tahminler.sort(key=lambda x: x[1], reverse=True)
        
        # En iyi n öneriyi döndür
        return tahminler[:n]
    
    def oneri_acikla(self, kullanici_id: int, film_id: int) -> str:
        """
        🎯 Bu metod ne yapar?
        Bir önerinin nedenini açıklar.
        
        İçerik tabanlı sistemlerin avantajı:
        "Neden bunu önerdin?" sorusuna cevap verebilir!
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - film_id (int): Önerilen film ID
        
        📤 Döndürdüğü:
        - aciklama (str): Açıklama metni
        """
        
        self.egitim_kontrolu()
        
        # Film bilgisini al
        film = self.filmler[self.filmler['movieId'] == film_id]
        if len(film) == 0:
            return "Film bulunamadı."
        
        film_adi = film['title'].values[0]
        film_turleri = film['genres'].values[0]
        
        # Kullanıcının favori türlerini bul
        if kullanici_id not in self.kullanici_profilleri:
            return "Kullanıcı bulunamadı."
        
        profil = self.kullanici_profilleri[kullanici_id]
        
        # En yüksek ağırlıklı türleri bul
        tur_agirliklar = list(zip(self.turler, profil))
        tur_agirliklar.sort(key=lambda x: x[1], reverse=True)
        favori_turler = [t[0] for t in tur_agirliklar[:3] if t[1] > 0]
        
        # Açıklama oluştur
        aciklama = f"📽️ '{film_adi}' önerisi\n"
        aciklama += f"   Türleri: {film_turleri}\n"
        aciklama += f"   Favori türleriniz: {', '.join(favori_turler)}\n"
        
        # Eşleşen türleri bul
        film_tur_listesi = film_turleri.split('|') if film_turleri != "(no genres listed)" else []
        eslesen = set(favori_turler) & set(film_tur_listesi)
        
        if eslesen:
            aciklama += f"   ✅ Eşleşen türler: {', '.join(eslesen)}"
        
        return aciklama


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 İçerik Tabanlı Model testi\n")
    
    # Test verisi oluştur
    filmler = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Film A', 'Film B', 'Film C', 'Film D', 'Film E'],
        'genres': ['Action|Adventure', 'Comedy|Romance', 'Action|Sci-Fi', 
                   'Drama|Romance', 'Action|Comedy']
    })
    
    puanlar = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2],
        'movieId': [1, 3, 5, 2, 4],
        'rating': [5.0, 4.5, 4.0, 5.0, 4.0]
    })
    
    # Model oluştur ve eğit
    model = ContentBasedModel()
    model.egit(filmler, puanlar)
    
    # Bilgi göster
    model.bilgi_yazdir()
    
    # Tahmin yap
    print("\n🔮 Tahminler:")
    print(f"   Kullanıcı 1, Film 2: {model.tahmin_et(1, 2):.2f}")
    print(f"   Kullanıcı 2, Film 1: {model.tahmin_et(2, 1):.2f}")
    
    # Öneri al
    print("\n🎬 Kullanıcı 1 için öneriler:")
    oneriler = model.oner(1, n=3)
    for film_id, tahmin in oneriler:
        print(f"   Film {film_id}: {tahmin:.2f}")
    
    print("\n✅ İçerik Tabanlı Model testi tamamlandı!")
