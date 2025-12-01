# ============================================
# 🏗️ Temel Model Sınıfı (Base Model)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Tüm öneri modelleri için temel sınıfı tanımlar.
# 
# "Abstract Base Class" (Soyut Temel Sınıf) nedir?
# Diğer sınıfların miras alacağı bir şablon.
# Kendi başına kullanılmaz, sadece temel kuralları belirler.
#
# Neden kullanıyoruz?
# Tüm modellerin aynı fonksiyonlara sahip olmasını
# garanti altına alıyoruz. Böylece kod tutarlı olur!
#
# 💡 Kullanım:
# Bu sınıftan miras alarak yeni modeller oluşturulur.
# ============================================

# ----- Gerekli Kütüphaneleri İçe Aktar -----

# ABC: Abstract Base Class (Soyut Temel Sınıf)
from abc import ABC, abstractmethod

# Tip belirtme için
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    🎯 Bu sınıf ne yapar?
    Tüm öneri modelleri için temel yapıyı tanımlar.
    
    Bu bir "soyut sınıf" - doğrudan kullanılmaz,
    diğer sınıflar bundan miras alır.
    
    Düşün ki bir araba şablonu var:
    - Her arabanın motoru olmalı
    - Her araba hareket edebilmeli
    
    Bu şablon gerçek bir araba değil, ama
    tüm arabaların nasıl olması gerektiğini söylüyor!
    
    📊 Özellikler:
    - isim: Model ismi
    - egitildi_mi: Model eğitilmiş mi?
    - filmler: Film bilgileri
    - puanlar: Puan bilgileri
    
    🔧 Metodlar:
    - egit(): Modeli eğit
    - tahmin_et(): Puan tahmini yap
    - oner(): Film öner
    """
    
    def __init__(self, isim: str = "BaseModel"):
        """
        🎯 Yapıcı (Constructor) metod.
        Yeni bir model oluşturulduğunda çalışır.
        
        📥 Parametreler:
        - isim (str): Model ismi
        
        💡 Örnek:
        >>> model = ContentBasedModel("İçerik Modeli")
        """
        
        # Model ismi
        # self.isim = isim demek: Bu nesnenin isim özelliği = parametre
        self.isim = isim
        
        # Model eğitildi mi? Başlangıçta hayır.
        self.egitildi_mi = False
        
        # Veri setleri (eğitim sırasında doldurulacak)
        self.filmler = None
        self.puanlar = None
        
        # Model parametreleri (her modelde farklı olabilir)
        self.parametreler = {}
        
        # Hoş geldin mesajı
        print(f"🎬 {self.isim} modeli oluşturuldu!")
    
    @abstractmethod
    def egit(self, filmler: pd.DataFrame, puanlar: pd.DataFrame) -> None:
        """
        🎯 Bu metod ne yapar?
        Modeli verilen verilerle eğitir.
        
        Bu bir SOYUT METOD! Yani:
        - Burada sadece tanım var, kod yok
        - Alt sınıflar BU METODU YAZMAK ZORUNDA
        
        📥 Parametreler:
        - filmler (DataFrame): Film tablosu
        - puanlar (DataFrame): Puan tablosu
        
        🤔 Neden soyut?
        Her modelin eğitim mantığı farklı.
        İçerik tabanlı başka, SVD başka çalışır.
        Bu metodu alt sınıflar kendileri yazacak.
        """
        # Bu kısım alt sınıfta doldurulacak
        pass
    
    @abstractmethod
    def tahmin_et(self, kullanici_id: int, film_id: int) -> float:
        """
        🎯 Bu metod ne yapar?
        Bir kullanıcının bir filme vereceği puanı tahmin eder.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID'si
        - film_id (int): Film ID'si
        
        📤 Döndürdüğü:
        - tahmin (float): Tahmini puan (0.5 - 5.0 arası)
        
        💡 Örnek:
        >>> tahmin = model.tahmin_et(kullanici_id=1, film_id=100)
        >>> print(f"Tahmini puan: {tahmin:.2f}")
        Tahmini puan: 4.25
        """
        pass
    
    @abstractmethod
    def oner(self, kullanici_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Kullanıcıya n tane film önerir.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID'si
        - n (int): Kaç film önerilecek
        
        📤 Döndürdüğü:
        - oneriler (list): [(film_id, tahmini_puan), ...] listesi
        
        💡 Örnek:
        >>> oneriler = model.oner(kullanici_id=1, n=5)
        >>> print(oneriler)
        [(100, 4.8), (205, 4.6), (42, 4.5), (189, 4.3), (77, 4.2)]
        """
        pass
    
    def egitim_kontrolu(self) -> None:
        """
        🎯 Bu metod ne yapar?
        Modelin eğitilip eğitilmediğini kontrol eder.
        Eğitilmediyse hata fırlatır.
        
        🤔 Neden bu metodu yazdık?
        Eğitilmemiş modelle tahmin yapmak hata verir.
        Bu kontrolle kullanıcıya net hata mesajı veririz.
        """
        
        if not self.egitildi_mi:
            raise RuntimeError(
                f"❌ {self.isim} henüz eğitilmedi!\n"
                f"   Önce model.egit(filmler, puanlar) çağırın."
            )
    
    def bilgi_yazdir(self) -> None:
        """
        🎯 Bu metod ne yapar?
        Model hakkında bilgi yazdırır.
        """
        
        print("\n" + "=" * 50)
        print(f"📊 MODEL BİLGİLERİ: {self.isim}")
        print("=" * 50)
        
        print(f"   Eğitildi mi: {'✅ Evet' if self.egitildi_mi else '❌ Hayır'}")
        
        if self.filmler is not None:
            print(f"   Film sayısı: {len(self.filmler):,}")
        
        if self.puanlar is not None:
            print(f"   Puan sayısı: {len(self.puanlar):,}")
        
        if self.parametreler:
            print(f"   Parametreler:")
            for anahtar, deger in self.parametreler.items():
                print(f"      {anahtar}: {deger}")
        
        print("=" * 50)
    
    def performans_olc(self, test_verisi: pd.DataFrame) -> dict:
        """
        🎯 Bu metod ne yapar?
        Model performansını test verisiyle ölçer.
        
        📥 Parametreler:
        - test_verisi (DataFrame): Test puan tablosu
          (userId, movieId, rating sütunları)
        
        📤 Döndürdüğü:
        - metrikler (dict): RMSE, MAE gibi metrik değerleri
        """
        
        # Eğitim kontrolü
        self.egitim_kontrolu()
        
        # Tahminleri topla
        gercek_puanlar = []
        tahminler = []
        
        print(f"🔄 {len(test_verisi)} tahmin yapılıyor...")
        
        for i, satir in test_verisi.iterrows():
            kullanici_id = satir['userId']
            film_id = satir['movieId']
            gercek_puan = satir['rating']
            
            try:
                tahmin = self.tahmin_et(kullanici_id, film_id)
                gercek_puanlar.append(gercek_puan)
                tahminler.append(tahmin)
            except Exception:
                # Tahmin yapılamayan durumları atla
                continue
        
        # Metrikleri hesapla
        from src.metrics import tum_metrikleri_hesapla
        
        metrikler = tum_metrikleri_hesapla(gercek_puanlar, tahminler)
        metrikler['tahmin_sayisi'] = len(tahminler)
        
        return metrikler
    
    def __str__(self) -> str:
        """
        🎯 Bu metod ne yapar?
        print(model) yapıldığında görünecek metin.
        """
        durum = "eğitildi" if self.egitildi_mi else "eğitilmedi"
        return f"📊 {self.isim} ({durum})"
    
    def __repr__(self) -> str:
        """
        🎯 Bu metod ne yapar?
        Model nesnesinin teknik temsili.
        """
        return f"{self.__class__.__name__}(isim='{self.isim}')"


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 BaseModel testi\n")
    
    # BaseModel soyut sınıf olduğu için doğrudan oluşturulamaz
    # Bu hata vermeli!
    try:
        model = BaseModel("Test")
    except TypeError as hata:
        print(f"✅ Beklenen hata: {hata}")
    
    print("\n✅ BaseModel testi tamamlandı!")
    print("   Bu sınıf doğrudan kullanılamaz,")
    print("   diğer modeller bundan miras alır.")
