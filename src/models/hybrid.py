# ============================================
# 🔀 Hibrit Model (Hybrid Recommender)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# İçerik tabanlı ve işbirlikçi filtrelemeyi birleştirir.
# 
# Hibrit Sistem nedir?
# Birden fazla öneri yöntemini birleştiren sistem.
# "Tek başına iyi, birlikte daha iyi!"
# 
# Neden hibrit kullanılır?
# Her yöntemin avantajları ve dezavantajları var.
# Birleştirerek dezavantajları azaltabiliriz!
# 
# Hibrit türleri:
# 1. Ağırlıklı (Weighted): Tahminlerin ağırlıklı ortalaması
# 2. Geçişli (Switching): Duruma göre bir yöntemi seç
# 3. Karışık (Mixed): Her yöntemin önerilerini birleştir
# 4. Basamaklı (Cascade): Bir yöntem filtrele, diğeri sırala
# 
# Bu sınıf "Ağırlıklı Hibrit" kullanır.
# 
# Avantajları:
# ✅ Cold Start problemini azaltır
# ✅ Daha dengeli öneriler
# ✅ Farklı senaryolara uyum sağlar
# 
# Dezavantajları:
# ❌ Daha karmaşık sistem
# ❌ Ağırlık ayarlaması gerekir
#
# 💡 Gerçek hayat örneği:
# Netflix: İçerik + İşbirlikçi + Zaman bazlı trendler
# Spotify: Ses özellikleri + Kullanıcı davranışları + Editör seçkileri
# ============================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_model import BaseModel
from .content_based import ContentBasedModel
from .collaborative import CollaborativeFilteringModel


class HybridModel(BaseModel):
    """
    🎯 Bu sınıf ne yapar?
    İçerik tabanlı ve işbirlikçi filtrelemeyi birleştirir.
    
    Her iki modelin tahminlerini ağırlıklı olarak birleştirir.
    
    📊 Parametreler:
    - icerik_agirligi: İçerik tabanlı modelin ağırlığı
    - isbirligi_agirligi: İşbirlikçi modelin ağırlığı
    
    💡 Kullanım:
    >>> model = HybridModel(icerik_agirligi=0.3, isbirligi_agirligi=0.7)
    >>> model.egit(filmler, puanlar)
    >>> oneriler = model.oner(kullanici_id=1, n=10)
    """
    
    def __init__(self,
                 isim: str = "Hibrit Model",
                 icerik_agirligi: float = 0.3,
                 isbirligi_agirligi: float = 0.7,
                 isbirligi_turu: str = "kullanici"):
        """
        🎯 Model nesnesini oluşturur.
        
        📥 Parametreler:
        - isim (str): Model ismi
        - icerik_agirligi (float): İçerik modeli ağırlığı (0-1)
        - isbirligi_agirligi (float): İşbirlikçi model ağırlığı (0-1)
        - isbirligi_turu (str): "kullanici" veya "film"
        
        NOT: icerik_agirligi + isbirligi_agirligi = 1 olmalı!
        """
        
        super().__init__(isim)
        
        # Ağırlıkları normalize et
        toplam = icerik_agirligi + isbirligi_agirligi
        self.icerik_agirligi = icerik_agirligi / toplam
        self.isbirligi_agirligi = isbirligi_agirligi / toplam
        
        # Alt modeller
        self.icerik_modeli = ContentBasedModel("İçerik (Hibrit)")
        self.isbirligi_modeli = CollaborativeFilteringModel(
            "İşbirlikçi (Hibrit)", 
            tur=isbirligi_turu
        )
        
        print(f"   Ağırlıklar:")
        print(f"      İçerik: {self.icerik_agirligi:.0%}")
        print(f"      İşbirlikçi: {self.isbirligi_agirligi:.0%}")
    
    def egit(self, filmler: pd.DataFrame, puanlar: pd.DataFrame) -> None:
        """
        🎯 Bu metod ne yapar?
        Her iki alt modeli de eğitir.
        
        📥 Parametreler:
        - filmler (DataFrame): Film tablosu
        - puanlar (DataFrame): Puan tablosu
        """
        
        print(f"🔄 {self.isim} eğitimi başlıyor...")
        print("=" * 50)
        
        # Verileri kaydet
        self.filmler = filmler.copy()
        self.puanlar = puanlar.copy()
        
        # ----- Alt Model 1: İçerik Tabanlı -----
        print("\n📝 1. İçerik Tabanlı Model:")
        self.icerik_modeli.egit(filmler, puanlar)
        
        # ----- Alt Model 2: İşbirlikçi -----
        print("\n👥 2. İşbirlikçi Filtreleme:")
        self.isbirligi_modeli.egit(filmler, puanlar)
        
        # Eğitim tamamlandı
        self.egitildi_mi = True
        
        self.parametreler = {
            'icerik_agirligi': self.icerik_agirligi,
            'isbirligi_agirligi': self.isbirligi_agirligi
        }
        
        print("\n" + "=" * 50)
        print(f"✅ {self.isim} eğitimi tamamlandı!")
    
    def tahmin_et(self, kullanici_id: int, film_id: int) -> float:
        """
        🎯 Bu metod ne yapar?
        İki modelin tahminlerini ağırlıklı olarak birleştirir.
        
        Formül:
        tahmin = w1 × içerik_tahmini + w2 × işbirlikçi_tahmini
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - film_id (int): Film ID
        
        📤 Döndürdüğü:
        - tahmin (float): Birleşik tahmini puan
        """
        
        self.egitim_kontrolu()
        
        # Her iki modelden tahmin al
        try:
            icerik_tahmin = self.icerik_modeli.tahmin_et(kullanici_id, film_id)
        except Exception:
            icerik_tahmin = self.puanlar['rating'].mean()
        
        try:
            isbirligi_tahmin = self.isbirligi_modeli.tahmin_et(kullanici_id, film_id)
        except Exception:
            isbirligi_tahmin = self.puanlar['rating'].mean()
        
        # Ağırlıklı ortalama
        tahmin = (
            self.icerik_agirligi * icerik_tahmin +
            self.isbirligi_agirligi * isbirligi_tahmin
        )
        
        # Aralığı kontrol et
        tahmin = max(0.5, min(5.0, tahmin))
        
        return float(tahmin)
    
    def oner(self, kullanici_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Kullanıcıya n tane film önerir.
        
        Her iki modelin önerilerini birleştirir.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - n (int): Kaç film önerilecek
        
        📤 Döndürdüğü:
        - oneriler (list): [(film_id, tahmin), ...] listesi
        """
        
        self.egitim_kontrolu()
        
        # Kullanıcının izlediği filmler
        izlenen = set(
            self.puanlar[self.puanlar['userId'] == kullanici_id]['movieId']
        )
        
        # Tüm filmler için hibrit tahmin yap
        tahminler = {}
        
        for film_id in self.filmler['movieId']:
            if film_id not in izlenen:
                tahmin = self.tahmin_et(kullanici_id, film_id)
                tahminler[film_id] = tahmin
        
        # Sırala ve döndür
        sirali = sorted(tahminler.items(), key=lambda x: x[1], reverse=True)
        
        return sirali[:n]
    
    def onerinin_kaynagini_analiz_et(self, 
                                      kullanici_id: int, 
                                      film_id: int) -> dict:
        """
        🎯 Bu metod ne yapar?
        Bir önerinin hangi modelden ne kadar etkilendiğini gösterir.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - film_id (int): Film ID
        
        📤 Döndürdüğü:
        - analiz (dict): Her modelin katkısı
        """
        
        self.egitim_kontrolu()
        
        # Her modelin tahmini
        icerik_tahmin = self.icerik_modeli.tahmin_et(kullanici_id, film_id)
        isbirligi_tahmin = self.isbirligi_modeli.tahmin_et(kullanici_id, film_id)
        hibrit_tahmin = self.tahmin_et(kullanici_id, film_id)
        
        # Film bilgisi
        film = self.filmler[self.filmler['movieId'] == film_id]
        film_adi = film['title'].values[0] if len(film) > 0 else "Bilinmiyor"
        
        analiz = {
            'film_id': film_id,
            'film_adi': film_adi,
            'icerik_tahmin': icerik_tahmin,
            'icerik_katki': self.icerik_agirligi * icerik_tahmin,
            'isbirligi_tahmin': isbirligi_tahmin,
            'isbirligi_katki': self.isbirligi_agirligi * isbirligi_tahmin,
            'hibrit_tahmin': hibrit_tahmin
        }
        
        return analiz
    
    def analizi_yazdir(self, analiz: dict):
        """
        🎯 Öneri analizini güzel formatta yazdırır.
        """
        
        print("\n" + "=" * 50)
        print(f"🔍 ÖNERİ ANALİZİ: {analiz['film_adi']}")
        print("=" * 50)
        
        print("\n📊 Model Tahminleri:")
        print(f"   📝 İçerik Tabanlı: {analiz['icerik_tahmin']:.2f}")
        print(f"      Ağırlık: {self.icerik_agirligi:.0%}")
        print(f"      Katkı: {analiz['icerik_katki']:.2f}")
        
        print(f"\n   👥 İşbirlikçi: {analiz['isbirligi_tahmin']:.2f}")
        print(f"      Ağırlık: {self.isbirligi_agirligi:.0%}")
        print(f"      Katkı: {analiz['isbirligi_katki']:.2f}")
        
        print(f"\n   🔀 Hibrit Sonuç: {analiz['hibrit_tahmin']:.2f}")
        
        # Hangi model daha etkili?
        if analiz['icerik_katki'] > analiz['isbirligi_katki']:
            print("\n   💡 Bu öneri daha çok İÇERİK benzerliğine dayanıyor.")
        else:
            print("\n   💡 Bu öneri daha çok KULLANıCı benzerliğine dayanıyor.")
        
        print("=" * 50)
    
    def cold_start_tespit_et(self, kullanici_id: int) -> str:
        """
        🎯 Bu metod ne yapar?
        Kullanıcının Cold Start durumunda olup olmadığını tespit eder.
        
        Cold Start: Yeni kullanıcı = Az veri = Kötü öneriler
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        
        📤 Döndürdüğü:
        - durum (str): Cold Start durumu açıklaması
        """
        
        kullanici_puanlari = self.puanlar[
            self.puanlar['userId'] == kullanici_id
        ]
        puan_sayisi = len(kullanici_puanlari)
        
        if puan_sayisi == 0:
            return "🥶 TAM COLD START: Kullanıcı hiç puan vermemiş!"
        elif puan_sayisi < 5:
            return f"❄️ COLD START: Sadece {puan_sayisi} puan. İçerik tabanlı ağırlıklı öner."
        elif puan_sayisi < 20:
            return f"🌤️ ILIK: {puan_sayisi} puan. Hibrit öneri uygun."
        else:
            return f"☀️ YETERLİ VERİ: {puan_sayisi} puan. İşbirlikçi modele güvenilebilir."
    
    def agirliklari_otomatik_ayarla(self, kullanici_id: int):
        """
        🎯 Bu metod ne yapar?
        Kullanıcının veri durumuna göre ağırlıkları ayarlar.
        
        Az veri → İçerik ağırlığı artar
        Çok veri → İşbirlikçi ağırlığı artar
        """
        
        kullanici_puanlari = self.puanlar[
            self.puanlar['userId'] == kullanici_id
        ]
        puan_sayisi = len(kullanici_puanlari)
        
        if puan_sayisi < 5:
            # Cold Start - İçeriğe ağırlık ver
            icerik = 0.8
            isbirligi = 0.2
        elif puan_sayisi < 20:
            # Ilık - Dengeli
            icerik = 0.4
            isbirligi = 0.6
        else:
            # Yeterli veri - İşbirlikçiye ağırlık ver
            icerik = 0.2
            isbirligi = 0.8
        
        print(f"🔧 Kullanıcı {kullanici_id} için ağırlıklar ayarlandı:")
        print(f"   Puan sayısı: {puan_sayisi}")
        print(f"   İçerik: {icerik:.0%}, İşbirlikçi: {isbirligi:.0%}")
        
        return icerik, isbirligi


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 Hibrit Model testi\n")
    
    # Test verisi oluştur
    filmler = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Film A', 'Film B', 'Film C', 'Film D', 'Film E'],
        'genres': ['Action|Adventure', 'Comedy|Romance', 'Action|Sci-Fi', 
                   'Drama|Romance', 'Action|Comedy']
    })
    
    puanlar = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3],
        'movieId': [1, 2, 3, 2, 3, 4, 1, 4],
        'rating': [5.0, 3.0, 4.5, 4.0, 5.0, 4.5, 4.0, 5.0]
    })
    
    # Model oluştur ve eğit
    model = HybridModel(icerik_agirligi=0.4, isbirligi_agirligi=0.6)
    model.egit(filmler, puanlar)
    
    # Cold Start durumu
    print("\n❄️ Cold Start Tespiti:")
    print(f"   Kullanıcı 1: {model.cold_start_tespit_et(1)}")
    print(f"   Kullanıcı 99: {model.cold_start_tespit_et(99)}")
    
    # Öneri al
    print("\n🎬 Kullanıcı 1 için öneriler:")
    oneriler = model.oner(1, n=3)
    for film_id, tahmin in oneriler:
        print(f"   Film {film_id}: {tahmin:.2f}")
    
    # Öneri analizi
    if oneriler:
        analiz = model.onerinin_kaynagini_analiz_et(1, oneriler[0][0])
        model.analizi_yazdir(analiz)
    
    print("\n✅ Hibrit Model testi tamamlandı!")
