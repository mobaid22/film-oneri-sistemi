# ============================================
# 🎬 Ana Öneri Motoru (Recommender Engine)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Tüm modelleri bir araya getiren ana sınıf.
# Film öneri sisteminin "beyin merkezi"!
#
# 💡 Kullanım:
# from src.recommender import FilmOneriSistemi
# sistem = FilmOneriSistemi()
# sistem.veri_yukle()
# sistem.model_egit("svd")
# oneriler = sistem.film_oner(kullanici_id=1)
# ============================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

# Modelleri içe aktar
from .models import (
    ContentBasedModel,
    CollaborativeFilteringModel,
    SVDModel,
    HybridModel
)

from .data_loader import veri_yukle, kullanici_film_matrisi_olustur
from .preprocessing import egitim_test_ayir
from .metrics import tum_metrikleri_hesapla, metrikleri_yazdir
from .utils import film_ara, populer_filmleri_bul, kullanici_profili_olustur


class FilmOneriSistemi:
    """
    🎯 Bu sınıf ne yapar?
    Film öneri sisteminin ana kontrol sınıfı.
    
    Tüm modelleri, veri yüklemeyi ve öneri
    üretmeyi tek bir yerden yönetir.
    
    Düşün ki bir restoranın baş şefi gibi:
    - Malzemeleri yönetir (veri)
    - Farklı tarifler kullanır (modeller)
    - En iyi yemeği sunar (öneriler)
    
    📊 Desteklenen Modeller:
    - "icerik": İçerik tabanlı filtreleme
    - "kullanici": Kullanıcı tabanlı işbirlikçi
    - "film": Film tabanlı işbirlikçi
    - "svd": SVD matris ayrıştırma
    - "hibrit": Hibrit sistem
    
    💡 Kullanım:
    >>> sistem = FilmOneriSistemi()
    >>> sistem.veri_yukle()
    >>> sistem.model_egit("hibrit")
    >>> oneriler = sistem.film_oner(kullanici_id=1)
    """
    
    def __init__(self):
        """
        🎯 Sistem nesnesini oluşturur.
        """
        
        print("=" * 60)
        print("🎬 FİLM ÖNERİ SİSTEMİ")
        print("=" * 60)
        
        # Veri setleri
        self.filmler = None
        self.puanlar = None
        self.egitim_seti = None
        self.test_seti = None
        
        # Mevcut model
        self.aktif_model = None
        self.aktif_model_adi = None
        
        # Tüm eğitilmiş modeller
        self.modeller = {}
        
        # Sistem durumu
        self.veri_yuklendi = False
        
        print("✅ Sistem başlatıldı!")
        print("   Şimdi veri_yukle() fonksiyonunu çağırın.")
    
    def veri_yukle(self, veri_yolu: Optional[str] = None) -> None:
        """
        🎯 Bu metod ne yapar?
        MovieLens veri setini yükler.
        
        📥 Parametreler:
        - veri_yolu (str, opsiyonel): Veri seti klasör yolu
        
        💡 Kullanım:
        >>> sistem.veri_yukle()
        veya
        >>> sistem.veri_yukle("/ozel/yol/ml-latest-small")
        """
        
        print("\n" + "-" * 50)
        print("📂 VERİ YÜKLEME")
        print("-" * 50)
        
        try:
            # Veriyi yükle
            self.filmler, self.puanlar = veri_yukle(veri_yolu)
            self.veri_yuklendi = True
            
            print("\n✅ Veri başarıyla yüklendi!")
            
        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            print("   Önce veri setini indirin:")
            print("   python data/download_data.py")
    
    def veri_bol(self, test_orani: float = 0.2) -> None:
        """
        🎯 Bu metod ne yapar?
        Veriyi eğitim ve test setlerine böler.
        
        📥 Parametreler:
        - test_orani (float): Test setinin oranı (0-1)
        """
        
        if not self.veri_yuklendi:
            print("❌ Önce veri_yukle() çağırın!")
            return
        
        print("\n" + "-" * 50)
        print("✂️ VERİ BÖLME")
        print("-" * 50)
        
        self.egitim_seti, self.test_seti = egitim_test_ayir(
            self.puanlar, 
            test_orani=test_orani
        )
    
    def model_egit(self, model_tipi: str = "hibrit", **kwargs) -> None:
        """
        🎯 Bu metod ne yapar?
        Seçilen model tipini eğitir.
        
        📥 Parametreler:
        - model_tipi (str): Model tipi
          - "icerik": İçerik tabanlı
          - "kullanici": Kullanıcı tabanlı işbirlikçi
          - "film": Film tabanlı işbirlikçi
          - "svd": SVD
          - "hibrit": Hibrit
        - **kwargs: Modele özel parametreler
        
        💡 Kullanım:
        >>> sistem.model_egit("svd", n_faktor=100, iterasyon=30)
        """
        
        if not self.veri_yuklendi:
            print("❌ Önce veri_yukle() çağırın!")
            return
        
        print("\n" + "-" * 50)
        print(f"🔧 MODEL EĞİTİMİ: {model_tipi.upper()}")
        print("-" * 50)
        
        # Model oluştur
        if model_tipi == "icerik":
            model = ContentBasedModel(**kwargs)
        
        elif model_tipi == "kullanici":
            model = CollaborativeFilteringModel(tur="kullanici", **kwargs)
        
        elif model_tipi == "film":
            model = CollaborativeFilteringModel(tur="film", **kwargs)
        
        elif model_tipi == "svd":
            model = SVDModel(**kwargs)
        
        elif model_tipi == "hibrit":
            model = HybridModel(**kwargs)
        
        else:
            print(f"❌ Bilinmeyen model tipi: {model_tipi}")
            print("   Geçerli tipler: icerik, kullanici, film, svd, hibrit")
            return
        
        # Veri setini seç (eğitim seti varsa onu kullan)
        if self.egitim_seti is not None:
            egitim_puanlar = self.egitim_seti
        else:
            egitim_puanlar = self.puanlar
        
        # Modeli eğit
        model.egit(self.filmler, egitim_puanlar)
        
        # Kaydet
        self.modeller[model_tipi] = model
        self.aktif_model = model
        self.aktif_model_adi = model_tipi
        
        print(f"\n✅ {model_tipi} modeli aktif!")
    
    def film_oner(self, 
                   kullanici_id: int, 
                   n: int = 10,
                   detayli: bool = True) -> List[Tuple[int, float]]:
        """
        🎯 Bu metod ne yapar?
        Kullanıcıya film önerir.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - n (int): Kaç film önerilecek
        - detayli (bool): Detaylı çıktı göster
        
        📤 Döndürdüğü:
        - oneriler (list): [(film_id, tahmin), ...] listesi
        
        💡 Kullanım:
        >>> oneriler = sistem.film_oner(kullanici_id=1, n=5)
        """
        
        if self.aktif_model is None:
            print("❌ Önce model_egit() çağırın!")
            return []
        
        # Öneri al
        oneriler = self.aktif_model.oner(kullanici_id, n)
        
        if detayli and oneriler:
            print(f"\n🎬 Kullanıcı {kullanici_id} için Öneriler:")
            print("-" * 50)
            
            for i, (film_id, tahmin) in enumerate(oneriler, 1):
                # Film bilgisini bul
                film = self.filmler[self.filmler['movieId'] == film_id]
                
                if len(film) > 0:
                    film_adi = film['title'].values[0]
                    turler = film['genres'].values[0]
                    
                    print(f"{i:2}. ⭐ {tahmin:.2f} | {film_adi}")
                    print(f"         Türler: {turler}")
        
        return oneriler
    
    def modeli_degerlendir(self) -> dict:
        """
        🎯 Bu metod ne yapar?
        Aktif modelin performansını test seti ile değerlendirir.
        
        📤 Döndürdüğü:
        - metrikler (dict): RMSE, MAE, vb.
        """
        
        if self.aktif_model is None:
            print("❌ Önce model_egit() çağırın!")
            return {}
        
        if self.test_seti is None:
            print("⚠️ Test seti yok. veri_bol() ile veriyi bölün.")
            # Test seti yoksa tüm veriyle değerlendir
            test = self.puanlar.sample(min(1000, len(self.puanlar)))
        else:
            test = self.test_seti
        
        print(f"\n📊 Model Değerlendirme: {self.aktif_model_adi}")
        print("-" * 50)
        
        metrikler = self.aktif_model.performans_olc(test)
        metrikleri_yazdir(metrikler)
        
        return metrikler
    
    def film_ara(self, arama: str, limit: int = 10) -> pd.DataFrame:
        """
        🎯 Film adına göre arama yapar.
        """
        
        if not self.veri_yuklendi:
            print("❌ Önce veri_yukle() çağırın!")
            return pd.DataFrame()
        
        return film_ara(self.filmler, arama, limit)
    
    def populer_filmler(self, limit: int = 10) -> pd.DataFrame:
        """
        🎯 En popüler filmleri listeler.
        """
        
        if not self.veri_yuklendi:
            print("❌ Önce veri_yukle() çağırın!")
            return pd.DataFrame()
        
        return populer_filmleri_bul(self.puanlar, self.filmler, limit=limit)
    
    def kullanici_profili(self, kullanici_id: int) -> dict:
        """
        🎯 Kullanıcı profilini gösterir.
        """
        
        if not self.veri_yuklendi:
            print("❌ Önce veri_yukle() çağırın!")
            return {}
        
        profil = kullanici_profili_olustur(
            kullanici_id, 
            self.puanlar, 
            self.filmler
        )
        
        # Profili yazdır
        from .utils import profili_yazdir
        profili_yazdir(profil)
        
        return profil
    
    def tum_modelleri_karsilastir(self) -> pd.DataFrame:
        """
        🎯 Bu metod ne yapar?
        Tüm modelleri eğitir ve karşılaştırır.
        
        📤 Döndürdüğü:
        - sonuclar (DataFrame): Her modelin metrikleri
        """
        
        if not self.veri_yuklendi:
            print("❌ Önce veri_yukle() çağırın!")
            return pd.DataFrame()
        
        # Veriyi böl
        if self.test_seti is None:
            self.veri_bol(test_orani=0.2)
        
        print("\n" + "=" * 60)
        print("🏆 MODEL KARŞILAŞTIRMASI")
        print("=" * 60)
        
        model_tipleri = ["icerik", "kullanici", "film", "svd", "hibrit"]
        sonuclar = []
        
        for model_tipi in model_tipleri:
            print(f"\n{'='*50}")
            
            # Eğit
            self.model_egit(model_tipi)
            
            # Değerlendir
            metrikler = self.modeli_degerlendir()
            
            if metrikler:
                sonuclar.append({
                    'Model': model_tipi,
                    'RMSE': metrikler.get('RMSE', np.nan),
                    'MAE': metrikler.get('MAE', np.nan),
                    'R2': metrikler.get('R2', np.nan)
                })
        
        # DataFrame oluştur
        df = pd.DataFrame(sonuclar)
        df = df.sort_values('RMSE')
        
        print("\n" + "=" * 60)
        print("📊 SONUÇ TABLOSU")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        
        # En iyi modeli seç
        if len(df) > 0:
            en_iyi = df.iloc[0]['Model']
            print(f"\n🏆 En iyi model: {en_iyi.upper()}")
            self.model_egit(en_iyi)
        
        return df
    
    def durum(self) -> None:
        """
        🎯 Sistem durumunu gösterir.
        """
        
        print("\n" + "=" * 50)
        print("📊 SİSTEM DURUMU")
        print("=" * 50)
        
        print(f"\n📂 Veri Durumu:")
        if self.veri_yuklendi:
            print(f"   ✅ Veri yüklendi")
            print(f"   🎬 Film sayısı: {len(self.filmler):,}")
            print(f"   ⭐ Puan sayısı: {len(self.puanlar):,}")
        else:
            print("   ❌ Veri yüklenmedi")
        
        print(f"\n🔧 Model Durumu:")
        if self.aktif_model:
            print(f"   ✅ Aktif model: {self.aktif_model_adi}")
        else:
            print("   ❌ Model eğitilmedi")
        
        if self.modeller:
            print(f"   📚 Eğitilmiş modeller: {list(self.modeller.keys())}")
        
        print("=" * 50)


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 Öneri Sistemi testi\n")
    
    # Sistem oluştur
    sistem = FilmOneriSistemi()
    
    # Test verisi oluştur (gerçek veri yüklemeden test için)
    sistem.filmler = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Film A', 'Film B', 'Film C', 'Film D', 'Film E'],
        'genres': ['Action', 'Comedy', 'Drama', 'Action|Comedy', 'Drama|Romance']
    })
    
    sistem.puanlar = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 2, 3, 3],
        'movieId': [1, 2, 3, 1, 3, 4, 2, 5],
        'rating': [5.0, 3.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0]
    })
    
    sistem.veri_yuklendi = True
    
    # Model eğit
    sistem.model_egit("icerik")
    
    # Öneri al
    sistem.film_oner(kullanici_id=1, n=3)
    
    # Durum göster
    sistem.durum()
    
    print("\n✅ Öneri Sistemi testi tamamlandı!")
