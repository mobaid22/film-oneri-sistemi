# ============================================
# 🔢 SVD Modeli (Singular Value Decomposition)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Matris ayrıştırma yöntemiyle öneri yapar.
# 
# SVD nedir?
# Büyük bir matrisi daha küçük parçalara ayırma tekniği.
# 
# Matematiksel gösterim:
# R ≈ U × Σ × V^T
# 
# R = Kullanıcı-Film puan matrisi (orijinal)
# U = Kullanıcı faktör matrisi (her kullanıcının gizli özellikleri)
# Σ = Tekil değerler (önem ağırlıkları)
# V^T = Film faktör matrisi (her filmin gizli özellikleri)
# 
# Gizli faktörler (Latent Factors) ne demek?
# Veride açıkça görünmeyen ama var olan özellikler.
# 
# Örnek:
# Faktör 1: "Aksiyon sevgisi" (0-1 arası)
# Faktör 2: "Romantizm sevgisi" (0-1 arası)
# Faktör 3: "Kara mizah beğenisi" (0-1 arası)
# 
# Bir kullanıcı: [0.9, 0.1, 0.6] → Aksiyon + Kara mizah sever
# Bir film: [0.8, 0.0, 0.7] → Aksiyon + Kara mizah
# Benzerlik yüksek → Öneri yapılır!
# 
# Avantajları:
# ✅ Büyük veri setlerinde verimli
# ✅ Gizli kalıpları keşfeder
# ✅ Seyreklik (sparsity) problemini azaltır
# 
# Dezavantajları:
# ❌ Yorumlaması zor (gizli faktörler ne demek?)
# ❌ Cold Start problemi var
# ❌ Eğitim süresi uzun olabilir
#
# 💡 Gerçek hayat örneği:
# Netflix Prize yarışmasının kazanan çözümü SVD tabanlıydı!
# ============================================

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from .base_model import BaseModel


class SVDModel(BaseModel):
    """
    🎯 Bu sınıf ne yapar?
    SVD (Singular Value Decomposition) ile öneri yapar.
    
    Kullanıcı ve film gizli faktörlerini öğrenir,
    bu faktörlerin çarpımıyla puan tahmin eder.
    
    📊 Parametreler:
    - n_faktor: Gizli faktör sayısı
    - ogrenme_orani: Gradient descent adım boyutu
    - regulasyon: Aşırı öğrenmeyi önleme katsayısı
    - iterasyon: Eğitim tekrar sayısı
    
    💡 Kullanım:
    >>> model = SVDModel(n_faktor=50)
    >>> model.egit(filmler, puanlar)
    >>> tahmin = model.tahmin_et(kullanici_id=1, film_id=100)
    """
    
    def __init__(self,
                 isim: str = "SVD Model",
                 n_faktor: int = 50,
                 ogrenme_orani: float = 0.005,
                 regulasyon: float = 0.02,
                 iterasyon: int = 20):
        """
        🎯 Model nesnesini oluşturur.
        
        📥 Parametreler:
        - isim (str): Model ismi
        - n_faktor (int): Gizli faktör sayısı
          Yüksek = Daha karmaşık kalıplar yakalanır
          Düşük = Daha basit ama genellenebilir
        - ogrenme_orani (float): Her adımda ne kadar güncelleme yapılacak
          Yüksek = Hızlı ama kararsız öğrenme
          Düşük = Yavaş ama kararlı öğrenme
        - regulasyon (float): Aşırı öğrenmeyi (overfitting) önleme
          Yüksek = Daha basit model
          Düşük = Veriye daha yakın model
        - iterasyon (int): Tüm veri üzerinden kaç geçiş yapılacak
        """
        
        super().__init__(isim)
        
        # Hiperparametreler
        self.n_faktor = n_faktor
        self.ogrenme_orani = ogrenme_orani
        self.regulasyon = regulasyon
        self.iterasyon = iterasyon
        
        # Öğrenilecek parametreler
        self.genel_ortalama = 0.0  # μ: Tüm puanların ortalaması
        self.kullanici_bias = None  # b_u: Kullanıcı eğilimleri
        self.film_bias = None  # b_i: Film eğilimleri
        self.P = None  # Kullanıcı faktör matrisi
        self.Q = None  # Film faktör matrisi
        
        # ID eşleşmeleri
        self.kullanici_id_to_idx = {}
        self.idx_to_kullanici_id = {}
        self.film_id_to_idx = {}
        self.idx_to_film_id = {}
        
        # Eğitim geçmişi
        self.egitim_gecmisi = []
        
        print(f"   Faktör sayısı: {self.n_faktor}")
        print(f"   Öğrenme oranı: {self.ogrenme_orani}")
        print(f"   Regulasyon: {self.regulasyon}")
        print(f"   İterasyon: {self.iterasyon}")
    
    def egit(self, filmler: pd.DataFrame, puanlar: pd.DataFrame) -> None:
        """
        🎯 Bu metod ne yapar?
        Modeli SGD (Stochastic Gradient Descent) ile eğitir.
        
        SGD nedir?
        Her veri noktası için hata hesapla ve
        parametreleri küçük adımlarla güncelle.
        
        Formül:
        r̂(u,i) = μ + b_u + b_i + P_u · Q_i
        
        Türkçesi:
        Tahmin = Genel ort. + Kullanıcı bias + Film bias + (Faktörler çarpımı)
        
        📥 Parametreler:
        - filmler (DataFrame): Film tablosu
        - puanlar (DataFrame): Puan tablosu
        """
        
        print(f"🔄 {self.isim} eğitimi başlıyor...")
        
        # Verileri kaydet
        self.filmler = filmler.copy()
        self.puanlar = puanlar.copy()
        
        # ----- Adım 1: ID Eşleşmelerini Oluştur -----
        kullanicilar = sorted(puanlar['userId'].unique())
        filmler_ids = sorted(puanlar['movieId'].unique())
        
        for idx, uid in enumerate(kullanicilar):
            self.kullanici_id_to_idx[uid] = idx
            self.idx_to_kullanici_id[idx] = uid
        
        for idx, mid in enumerate(filmler_ids):
            self.film_id_to_idx[mid] = idx
            self.idx_to_film_id[idx] = mid
        
        n_kullanici = len(kullanicilar)
        n_film = len(filmler_ids)
        
        print(f"   📊 {n_kullanici:,} kullanıcı, {n_film:,} film")
        
        # ----- Adım 2: Parametreleri Başlat -----
        # Genel ortalama
        self.genel_ortalama = puanlar['rating'].mean()
        
        # Bias'ları sıfırla
        self.kullanici_bias = np.zeros(n_kullanici)
        self.film_bias = np.zeros(n_film)
        
        # Faktör matrislerini küçük rastgele değerlerle başlat
        # Normal dağılım, ortalama=0, std=0.1
        self.P = np.random.normal(0, 0.1, (n_kullanici, self.n_faktor))
        self.Q = np.random.normal(0, 0.1, (n_film, self.n_faktor))
        
        # ----- Adım 3: SGD ile Eğit -----
        print(f"   🔄 {self.iterasyon} iterasyon eğitim...")
        
        for epoch in range(self.iterasyon):
            # Veriyi karıştır (her epoch farklı sıra)
            puanlar_karisik = puanlar.sample(frac=1, random_state=epoch)
            
            toplam_hata = 0
            
            for _, satir in puanlar_karisik.iterrows():
                u_idx = self.kullanici_id_to_idx[satir['userId']]
                i_idx = self.film_id_to_idx[satir['movieId']]
                gercek = satir['rating']
                
                # Tahmin yap
                tahmin = self._tahmin_hesapla(u_idx, i_idx)
                
                # Hata hesapla
                hata = gercek - tahmin
                toplam_hata += hata ** 2
                
                # Gradyanları hesapla ve güncelle
                # Bias güncellemeleri
                self.kullanici_bias[u_idx] += self.ogrenme_orani * (
                    hata - self.regulasyon * self.kullanici_bias[u_idx]
                )
                self.film_bias[i_idx] += self.ogrenme_orani * (
                    hata - self.regulasyon * self.film_bias[i_idx]
                )
                
                # Faktör güncellemeleri
                P_u = self.P[u_idx].copy()
                Q_i = self.Q[i_idx].copy()
                
                self.P[u_idx] += self.ogrenme_orani * (
                    hata * Q_i - self.regulasyon * P_u
                )
                self.Q[i_idx] += self.ogrenme_orani * (
                    hata * P_u - self.regulasyon * Q_i
                )
            
            # Epoch sonucu
            rmse = np.sqrt(toplam_hata / len(puanlar))
            self.egitim_gecmisi.append(rmse)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      Epoch {epoch + 1}/{self.iterasyon}: RMSE = {rmse:.4f}")
        
        # Eğitim tamamlandı
        self.egitildi_mi = True
        
        self.parametreler = {
            'n_faktor': self.n_faktor,
            'ogrenme_orani': self.ogrenme_orani,
            'regulasyon': self.regulasyon,
            'iterasyon': self.iterasyon,
            'son_rmse': self.egitim_gecmisi[-1]
        }
        
        print(f"✅ {self.isim} eğitimi tamamlandı!")
        print(f"   Son RMSE: {self.egitim_gecmisi[-1]:.4f}")
    
    def _tahmin_hesapla(self, u_idx: int, i_idx: int) -> float:
        """
        🎯 İç tahmin hesaplama (indekslerle).
        
        Formül: r̂(u,i) = μ + b_u + b_i + P_u · Q_i
        """
        
        tahmin = (
            self.genel_ortalama +
            self.kullanici_bias[u_idx] +
            self.film_bias[i_idx] +
            np.dot(self.P[u_idx], self.Q[i_idx])
        )
        
        # Aralığı sınırla
        tahmin = max(0.5, min(5.0, tahmin))
        
        return tahmin
    
    def tahmin_et(self, kullanici_id: int, film_id: int) -> float:
        """
        🎯 Bu metod ne yapar?
        Kullanıcının filme vereceği puanı tahmin eder.
        
        📥 Parametreler:
        - kullanici_id (int): Kullanıcı ID
        - film_id (int): Film ID
        
        📤 Döndürdüğü:
        - tahmin (float): Tahmini puan
        """
        
        self.egitim_kontrolu()
        
        # Bilinmeyen kullanıcı/film için ortalama döndür
        if kullanici_id not in self.kullanici_id_to_idx:
            return self.genel_ortalama
        
        if film_id not in self.film_id_to_idx:
            return self.genel_ortalama
        
        u_idx = self.kullanici_id_to_idx[kullanici_id]
        i_idx = self.film_id_to_idx[film_id]
        
        return self._tahmin_hesapla(u_idx, i_idx)
    
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
        
        if kullanici_id not in self.kullanici_id_to_idx:
            print(f"⚠️ Kullanıcı {kullanici_id} bulunamadı!")
            return []
        
        # Kullanıcının puanladığı filmler
        izlenen = set(
            self.puanlar[self.puanlar['userId'] == kullanici_id]['movieId']
        )
        
        # Tüm filmler için tahmin yap
        tahminler = []
        
        for film_id in self.film_id_to_idx.keys():
            if film_id not in izlenen:
                tahmin = self.tahmin_et(kullanici_id, film_id)
                tahminler.append((film_id, tahmin))
        
        # Puana göre sırala
        tahminler.sort(key=lambda x: x[1], reverse=True)
        
        return tahminler[:n]
    
    def faktor_analizi(self, n_top: int = 5):
        """
        🎯 Bu metod ne yapar?
        En önemli gizli faktörleri analiz eder.
        
        Her faktör için:
        - En yüksek değere sahip kullanıcılar
        - En yüksek değere sahip filmler
        
        Bu faktörlerin "ne anlama geldiğini" anlamaya yardımcı olur!
        """
        
        self.egitim_kontrolu()
        
        print("\n" + "=" * 50)
        print("🔍 FAKTÖR ANALİZİ")
        print("=" * 50)
        
        # İlk 3 faktörü analiz et
        for f in range(min(3, self.n_faktor)):
            print(f"\n📊 Faktör {f + 1}:")
            
            # Bu faktörde en yüksek kullanıcılar
            kullanici_skorlari = [(self.idx_to_kullanici_id[i], self.P[i, f]) 
                                   for i in range(len(self.P))]
            kullanici_skorlari.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   👤 En yüksek kullanıcılar:")
            for uid, skor in kullanici_skorlari[:n_top]:
                print(f"      Kullanıcı {uid}: {skor:.3f}")
            
            # Bu faktörde en yüksek filmler
            film_skorlari = [(self.idx_to_film_id[i], self.Q[i, f]) 
                            for i in range(len(self.Q))]
            film_skorlari.sort(key=lambda x: x[1], reverse=True)
            
            print(f"   🎬 En yüksek filmler:")
            for mid, skor in film_skorlari[:n_top]:
                # Film adını bul
                film = self.filmler[self.filmler['movieId'] == mid]
                if len(film) > 0:
                    film_adi = film['title'].values[0][:40]
                    print(f"      {film_adi}: {skor:.3f}")
        
        print("=" * 50)
    
    def egitim_grafigi_ciz(self):
        """
        🎯 Bu metod ne yapar?
        Eğitim sürecindeki RMSE değişimini gösterir.
        """
        
        if not self.egitim_gecmisi:
            print("⚠️ Henüz eğitim yapılmadı!")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.egitim_gecmisi) + 1), 
                    self.egitim_gecmisi, 'b-', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('RMSE', fontsize=12)
            plt.title('SVD Eğitim Süreci', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("⚠️ matplotlib yüklü değil, grafik çizilemedi.")
            print("   Eğitim geçmişi:", self.egitim_gecmisi)


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 SVD Model testi\n")
    
    # Test verisi oluştur
    np.random.seed(42)
    
    puanlar = pd.DataFrame({
        'userId': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        'movieId': [1, 2, 3, 4, 1, 2, 5, 2, 3, 4, 5, 1, 3, 5],
        'rating': [5.0, 4.0, 3.0, 5.0, 4.0, 5.0, 3.0, 3.0, 4.0, 5.0, 4.0, 5.0, 4.0, 4.0]
    })
    
    filmler = pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5],
        'title': ['Film A', 'Film B', 'Film C', 'Film D', 'Film E'],
        'genres': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy']
    })
    
    # Model oluştur ve eğit
    model = SVDModel(n_faktor=5, iterasyon=10)
    model.egit(filmler, puanlar)
    
    # Bilgi göster
    model.bilgi_yazdir()
    
    # Tahmin yap
    print("\n🔮 Tahminler:")
    print(f"   Kullanıcı 1, Film 5: {model.tahmin_et(1, 5):.2f}")
    print(f"   Kullanıcı 2, Film 3: {model.tahmin_et(2, 3):.2f}")
    
    # Öneri al
    print("\n🎬 Kullanıcı 1 için öneriler:")
    oneriler = model.oner(1, n=3)
    for film_id, tahmin in oneriler:
        print(f"   Film {film_id}: {tahmin:.2f}")
    
    print("\n✅ SVD Model testi tamamlandı!")
