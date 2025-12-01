# ============================================
# 📊 Metrik Hesaplama Modülü (Metrics)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Öneri sisteminin performansını ölçen metrikleri hesaplar.
# 
# Metrik nedir?
# Modelimizin ne kadar iyi tahmin yaptığını
# ölçen sayısal değerler. Bir öğrencinin sınav
# notu gibi düşün - ne kadar yüksek/düşük olursa
# o kadar iyi/kötü!
#
# 💡 Kullanım:
# from src.metrics import rmse_hesapla, mae_hesapla
# ============================================

# ----- Gerekli Kütüphaneleri İçe Aktar -----

import numpy as np
from typing import Union, List


# ============================================
# 📏 RMSE - Root Mean Square Error
# Türkçe: Kök Ortalama Kare Hata
# ============================================
# 
# Formül: √(1/n × Σ(gerçek - tahmin)²)
# 
# Adım adım açıklama:
# 1. Her tahmin için hatayı bul: (gerçek - tahmin)
# 2. Hataların karesini al: hata²
# 3. Karelerin ortalamasını hesapla: Σ(hata²) / n
# 4. Karekökünü al: √(ortalama)
# 
# Ne işe yarar?
# Tahminlerimizin gerçek değerlerden ne kadar uzak
# olduğunu ölçer. Sonuç ne kadar KÜÇÜK olursa,
# tahminlerimiz o kadar İYİ demektir.
# 
# Neden kare alıyoruz?
# 1. Negatif hataları pozitif yapmak için
# 2. Büyük hataları daha çok cezalandırmak için
#    (2 hata → 4, ama 4 hata → 16!)
# 
# Gerçek hayat örneği:
# Netflix film puanı tahmini yapıyor.
# Gerçek puan: 4, Tahmin: 3.5 → Hata: 0.5
# Gerçek puan: 5, Tahmin: 3.0 → Hata: 2.0
# RMSE bize ortalama hatayı söyler.
# ============================================

def rmse_hesapla(gercek: Union[np.ndarray, List], tahmin: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    RMSE (Kök Ortalama Kare Hata) metriğini hesaplar.
    
    RMSE, tahmin hatalarımızın "ortalama büyüklüğünü" ölçer.
    Büyük hataları daha çok cezalandırır!
    
    📥 Parametreler:
    - gercek: Gerçek değerler listesi
      Örnek: [5.0, 4.0, 3.0, 5.0]
    - tahmin: Tahmin edilen değerler listesi
      Örnek: [4.5, 4.0, 3.5, 4.0]
    
    📤 Döndürdüğü:
    - rmse (float): RMSE değeri (0'a yakın = iyi)
    
    💡 Örnek Kullanım:
    >>> gercek = [5, 4, 3, 5]
    >>> tahmin = [4.5, 4.0, 3.5, 4.0]
    >>> rmse = rmse_hesapla(gercek, tahmin)
    >>> print(f"RMSE: {rmse:.4f}")
    RMSE: 0.6124
    
    🤔 Neden bu fonksiyonu yazdık?
    RMSE, öneri sistemlerinde en çok kullanılan metriktir!
    Netflix Prize yarışmasında bile bu metrik kullanıldı.
    """
    
    # Numpy dizisine çevir (liste olarak gelebilir)
    gercek = np.array(gercek)
    tahmin = np.array(tahmin)
    
    # Boyut kontrolü
    if len(gercek) != len(tahmin):
        raise ValueError(
            f"❌ Boyut uyuşmazlığı! "
            f"Gerçek: {len(gercek)}, Tahmin: {len(tahmin)}"
        )
    
    # Adım 1: Hataları hesapla (gerçek - tahmin)
    # Örnek: [5, 4] - [4.5, 4] = [0.5, 0]
    hatalar = gercek - tahmin
    
    # Adım 2: Hataların karesini al
    # Örnek: [0.5, 0]² = [0.25, 0]
    kare_hatalar = hatalar ** 2
    
    # Adım 3: Ortalamasını al
    # Örnek: (0.25 + 0) / 2 = 0.125
    ortalama_kare = np.mean(kare_hatalar)
    
    # Adım 4: Karekökünü al
    # Örnek: √0.125 = 0.3535
    rmse = np.sqrt(ortalama_kare)
    
    return float(rmse)


# ============================================
# 📏 MAE - Mean Absolute Error
# Türkçe: Ortalama Mutlak Hata
# ============================================
# 
# Formül: (1/n) × Σ|gerçek - tahmin|
# 
# Ne işe yarar?
# RMSE'ye benzer ama büyük hataları ekstra cezalandırmaz.
# Daha "adil" bir ortalama verir.
# 
# RMSE vs MAE:
# - RMSE: Büyük hataları daha çok cezalandırır
# - MAE: Tüm hatalara eşit davranır
# 
# Hangisini kullanalım?
# - Büyük hataları önlemek istiyorsan: RMSE
# - Genel performansı ölçmek istiyorsan: MAE
# ============================================

def mae_hesapla(gercek: Union[np.ndarray, List], tahmin: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    MAE (Ortalama Mutlak Hata) metriğini hesaplar.
    
    Her hatanın mutlak değerini alıp ortalamasını hesaplar.
    Mutlak değer = Negatif sayıları pozitif yapar
    |-3| = 3, |5| = 5
    
    📥 Parametreler:
    - gercek: Gerçek değerler listesi
    - tahmin: Tahmin edilen değerler listesi
    
    📤 Döndürdüğü:
    - mae (float): MAE değeri (0'a yakın = iyi)
    
    💡 Örnek Kullanım:
    >>> gercek = [5, 4, 3, 5]
    >>> tahmin = [4.5, 4.0, 3.5, 4.0]
    >>> mae = mae_hesapla(gercek, tahmin)
    >>> print(f"MAE: {mae:.4f}")
    MAE: 0.5000
    
    🤔 Neden bu fonksiyonu yazdık?
    MAE daha kolay yorumlanır! MAE = 0.5 demek,
    ortalama 0.5 puan hata yapıyoruz demek.
    """
    
    gercek = np.array(gercek)
    tahmin = np.array(tahmin)
    
    if len(gercek) != len(tahmin):
        raise ValueError(f"❌ Boyut uyuşmazlığı!")
    
    # Hataları hesapla
    hatalar = gercek - tahmin
    
    # Mutlak değerlerini al
    # np.abs: Absolute (mutlak) değer
    mutlak_hatalar = np.abs(hatalar)
    
    # Ortalamasını al
    mae = np.mean(mutlak_hatalar)
    
    return float(mae)


# ============================================
# 📏 MSE - Mean Square Error
# Türkçe: Ortalama Kare Hata
# ============================================
# 
# Formül: (1/n) × Σ(gerçek - tahmin)²
# 
# RMSE ile farkı:
# MSE = RMSE² (karekök almadan)
# RMSE = √MSE (karekök alarak)
# 
# Neden MSE kullanılır?
# Matematiksel hesaplamalarda daha kolay (türev almak gibi)
# ============================================

def mse_hesapla(gercek: Union[np.ndarray, List], tahmin: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    MSE (Ortalama Kare Hata) metriğini hesaplar.
    
    RMSE'nin karesiz hali. Matematiksel işlemlerde
    (örneğin gradient descent'te) daha kullanışlı.
    
    📥 Parametreler:
    - gercek: Gerçek değerler
    - tahmin: Tahmin edilen değerler
    
    📤 Döndürdüğü:
    - mse (float): MSE değeri
    
    💡 Örnek Kullanım:
    >>> mse = mse_hesapla([5, 4], [4.5, 3.5])
    >>> print(f"MSE: {mse}")
    MSE: 0.25
    """
    
    gercek = np.array(gercek)
    tahmin = np.array(tahmin)
    
    if len(gercek) != len(tahmin):
        raise ValueError(f"❌ Boyut uyuşmazlığı!")
    
    # Hata kareleri ortalaması
    mse = np.mean((gercek - tahmin) ** 2)
    
    return float(mse)


# ============================================
# 📏 R² - R-Squared (Belirlilik Katsayısı)
# Türkçe: R-Kare veya Determinasyon Katsayısı
# ============================================
# 
# Formül: 1 - (SS_res / SS_tot)
# SS_res = Σ(gerçek - tahmin)² (Kalıntı kareler toplamı)
# SS_tot = Σ(gerçek - ortalama)² (Toplam kareler toplamı)
# 
# Ne anlama gelir?
# Modelimizin veriyi ne kadar iyi açıkladığını gösterir.
# 
# Değer aralığı:
# - R² = 1: Mükemmel tahmin (%100)
# - R² = 0: Model ortalama kadar iyi (faydasız)
# - R² < 0: Model ortalamadan kötü!
# 
# Gerçek hayat örneği:
# R² = 0.85 → Modelimiz değişkenliğin %85'ini açıklıyor
# ============================================

def r2_hesapla(gercek: Union[np.ndarray, List], tahmin: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    R² (Belirlilik Katsayısı) metriğini hesaplar.
    
    Modelimizin veriyi ne kadar iyi "yakaladığını" ölçer.
    1'e yakın = çok iyi, 0 veya negatif = kötü
    
    📥 Parametreler:
    - gercek: Gerçek değerler
    - tahmin: Tahmin edilen değerler
    
    📤 Döndürdüğü:
    - r2 (float): R² değeri (-∞ ile 1 arası, 1'e yakın = iyi)
    
    💡 Örnek Kullanım:
    >>> r2 = r2_hesapla([3, 4, 5, 6], [3.1, 3.9, 5.2, 5.8])
    >>> print(f"R²: {r2:.4f}")
    R²: 0.9600
    
    🤔 Neden bu fonksiyonu yazdık?
    RMSE bize hatanın büyüklüğünü söyler.
    R² ise modelin ne kadar "iyi" olduğunu söyler.
    İkisi birlikte değerlendirilmeli!
    """
    
    gercek = np.array(gercek)
    tahmin = np.array(tahmin)
    
    if len(gercek) != len(tahmin):
        raise ValueError(f"❌ Boyut uyuşmazlığı!")
    
    # Gerçek değerlerin ortalaması
    ortalama = np.mean(gercek)
    
    # SS_res: Kalıntı kareler toplamı (Residual Sum of Squares)
    # Model tahminlerinin gerçekten sapması
    ss_res = np.sum((gercek - tahmin) ** 2)
    
    # SS_tot: Toplam kareler toplamı (Total Sum of Squares)
    # Gerçek değerlerin ortalamadan sapması
    ss_tot = np.sum((gercek - ortalama) ** 2)
    
    # R² hesapla
    # ss_tot 0 ise (tüm değerler aynı), 0 döndür
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    
    return float(r2)


# ============================================
# 📏 MAPE - Mean Absolute Percentage Error
# Türkçe: Ortalama Mutlak Yüzde Hata
# ============================================
# 
# Formül: (1/n) × Σ|((gerçek - tahmin) / gerçek)| × 100
# 
# Ne işe yarar?
# Hatayı yüzde olarak gösterir.
# "Ortalama %X hata yapıyoruz" diyebiliriz.
# 
# Dikkat:
# Gerçek değer 0 olduğunda hesaplanamaz (sıfıra bölme!)
# ============================================

def mape_hesapla(gercek: Union[np.ndarray, List], tahmin: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    MAPE (Ortalama Mutlak Yüzde Hata) metriğini hesaplar.
    
    Hatayı yüzde olarak ifade eder.
    "Tahminlerimiz ortalama %10 hatalı" gibi!
    
    📥 Parametreler:
    - gercek: Gerçek değerler (sıfır içermemeli!)
    - tahmin: Tahmin edilen değerler
    
    📤 Döndürdüğü:
    - mape (float): MAPE değeri (yüzde olarak)
    
    💡 Örnek Kullanım:
    >>> mape = mape_hesapla([100, 200], [90, 210])
    >>> print(f"MAPE: %{mape:.2f}")
    MAPE: %7.50
    
    ⚠️ Uyarı:
    Gerçek değerler 0 içeriyorsa hata verir!
    Film puanları genelde 0.5-5 arası olduğu için sorun olmaz.
    """
    
    gercek = np.array(gercek, dtype=float)
    tahmin = np.array(tahmin)
    
    if len(gercek) != len(tahmin):
        raise ValueError(f"❌ Boyut uyuşmazlığı!")
    
    # Sıfır kontrolü
    if np.any(gercek == 0):
        raise ValueError("❌ Gerçek değerler sıfır içeremez (sıfıra bölme!)")
    
    # Yüzde hataları hesapla
    yuzde_hatalar = np.abs((gercek - tahmin) / gercek) * 100
    
    # Ortalamasını al
    mape = np.mean(yuzde_hatalar)
    
    return float(mape)


# ============================================
# 📊 Precision, Recall, F1-Score
# Öneri Sistemleri İçin Özel Metrikler
# ============================================
# 
# Bu metrikler "önerdiğimiz filmler ne kadar isabetli?"
# sorusuna cevap verir.
# 
# Precision (Kesinlik): Önerdiğimiz filmlerin kaçı doğru?
# Recall (Duyarlılık): Doğru filmlerin kaçını önerdik?
# F1-Score: Precision ve Recall'ın dengeli ortalaması
# ============================================

def precision_hesapla(onerilen: set, gercek_begenilenler: set) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    Precision (Kesinlik) metriğini hesaplar.
    
    Soru: Önerdiğimiz filmlerin kaçı gerçekten beğenilecek?
    
    Formül: Doğru Öneriler / Toplam Öneriler
    
    📥 Parametreler:
    - onerilen (set): Önerdiğimiz film ID'leri
      Örnek: {1, 5, 10, 15, 20}
    - gercek_begenilenler (set): Kullanıcının gerçekten beğendiği filmler
      Örnek: {1, 5, 8, 12, 20}
    
    📤 Döndürdüğü:
    - precision (float): 0-1 arası (1 = mükemmel)
    
    💡 Örnek Kullanım:
    >>> onerilen = {1, 5, 10, 15, 20}
    >>> begenilenler = {1, 5, 8, 12, 20}
    >>> p = precision_hesapla(onerilen, begenilenler)
    >>> print(f"Precision: {p:.2f}")  # 3/5 = 0.60
    Precision: 0.60
    
    🤔 Gerçek hayat örneği:
    Netflix 5 film önerdi: [Film1, Film5, Film10, Film15, Film20]
    Kullanıcı bunlardan 3'ünü beğendi: [Film1, Film5, Film20]
    Precision = 3/5 = %60 isabetli öneri
    """
    
    # Set'e çevir (liste olarak gelebilir)
    onerilen = set(onerilen)
    gercek_begenilenler = set(gercek_begenilenler)
    
    # Hiç öneri yoksa, 0 döndür
    if len(onerilen) == 0:
        return 0.0
    
    # Kesişim: Hem önerilen hem beğenilen filmler
    # & operatörü: Set kesişimi
    dogru_oneriler = onerilen & gercek_begenilenler
    
    # Precision = Doğru Öneriler / Toplam Öneriler
    precision = len(dogru_oneriler) / len(onerilen)
    
    return float(precision)


def recall_hesapla(onerilen: set, gercek_begenilenler: set) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    Recall (Duyarlılık) metriğini hesaplar.
    
    Soru: Beğenilecek filmlerin kaçını önerdik?
    
    Formül: Doğru Öneriler / Toplam Beğenilenler
    
    📥 Parametreler:
    - onerilen (set): Önerdiğimiz film ID'leri
    - gercek_begenilenler (set): Kullanıcının beğendiği tüm filmler
    
    📤 Döndürdüğü:
    - recall (float): 0-1 arası (1 = mükemmel)
    
    💡 Örnek Kullanım:
    >>> onerilen = {1, 5, 10}
    >>> begenilenler = {1, 5, 8, 12, 20}
    >>> r = recall_hesapla(onerilen, begenilenler)
    >>> print(f"Recall: {r:.2f}")  # 2/5 = 0.40
    Recall: 0.40
    
    🤔 Precision vs Recall:
    - Precision: "Önerilerim ne kadar isabetli?"
    - Recall: "Beğenilecek filmlerin ne kadarını yakaladım?"
    
    İkisi arasında denge önemli!
    - Sadece 1 film önersek → Precision yüksek, Recall düşük
    - 1000 film önersek → Recall yüksek, Precision düşük
    """
    
    onerilen = set(onerilen)
    gercek_begenilenler = set(gercek_begenilenler)
    
    # Beğenilen film yoksa, 0 döndür
    if len(gercek_begenilenler) == 0:
        return 0.0
    
    # Kesişim: Hem önerilen hem beğenilen
    dogru_oneriler = onerilen & gercek_begenilenler
    
    # Recall = Doğru Öneriler / Toplam Beğenilenler
    recall = len(dogru_oneriler) / len(gercek_begenilenler)
    
    return float(recall)


def f1_hesapla(precision: float, recall: float) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    F1-Score metriğini hesaplar.
    
    F1-Score, Precision ve Recall'ın harmonik ortalamasıdır.
    İkisi arasında denge kurar.
    
    Formül: 2 × (Precision × Recall) / (Precision + Recall)
    
    Neden harmonik ortalama?
    Normal ortalama: (0.9 + 0.1) / 2 = 0.5 (yanıltıcı!)
    Harmonik ortalama: 2 × (0.9 × 0.1) / (0.9 + 0.1) = 0.18 (daha gerçekçi)
    
    Harmonik ortalama, düşük değerleri daha çok cezalandırır.
    
    📥 Parametreler:
    - precision (float): Kesinlik değeri
    - recall (float): Duyarlılık değeri
    
    📤 Döndürdüğü:
    - f1 (float): F1-Score (0-1 arası)
    
    💡 Örnek Kullanım:
    >>> f1 = f1_hesapla(precision=0.8, recall=0.6)
    >>> print(f"F1: {f1:.2f}")
    F1: 0.69
    """
    
    # İkisi de 0 ise, F1 = 0
    if precision + recall == 0:
        return 0.0
    
    # Harmonik ortalama formülü
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return float(f1)


def ndcg_hesapla(onerilen_sirali: list, gercek_ilgi: dict, k: int = 10) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    NDCG (Normalized Discounted Cumulative Gain) hesaplar.
    
    NDCG, sıralama kalitesini ölçer.
    "Doğru filmler listenin başında mı?" sorusuna cevap verir.
    
    Neden önemli?
    Bir kullanıcı genelde ilk 5-10 öneriye bakar.
    İyi filmler altta kalırsa, kimse görmez!
    
    📥 Parametreler:
    - onerilen_sirali (list): Sıralı film ID listesi [en iyi, ..., en kötü]
    - gercek_ilgi (dict): Film ID → İlgi puanı eşleşmesi
      Örnek: {1: 5, 5: 4, 10: 3} (Film 1'e ilgi = 5)
    - k (int): İlk kaç öneriye bakılacak
    
    📤 Döndürdüğü:
    - ndcg (float): 0-1 arası (1 = mükemmel sıralama)
    
    💡 Örnek Kullanım:
    >>> oneriler = [1, 5, 10, 15]
    >>> ilgiler = {1: 5, 5: 3, 10: 2, 20: 5}
    >>> ndcg = ndcg_hesapla(oneriler, ilgiler, k=3)
    """
    
    # DCG hesapla (Discounted Cumulative Gain)
    def dcg_hesapla(ilgi_listesi):
        """
        DCG = Σ (ilgi_i / log2(i + 1))
        Sıradaki pozisyon arttıkça, katkı azalır (discount)
        """
        dcg = 0.0
        for i, ilgi in enumerate(ilgi_listesi):
            # log2(2) = 1, log2(3) = 1.58, log2(4) = 2 ...
            # Pozisyon arttıkça bölen büyür, katkı küçülür
            dcg += ilgi / np.log2(i + 2)  # +2 çünkü log2(1) = 0
        return dcg
    
    # İlk k öneriyi al
    onerilen_k = onerilen_sirali[:k]
    
    # Önerilen filmlerin ilgi puanlarını al
    ilgi_listesi = []
    for film_id in onerilen_k:
        # Film ilgi sözlüğünde varsa puanını al, yoksa 0
        ilgi = gercek_ilgi.get(film_id, 0)
        ilgi_listesi.append(ilgi)
    
    # DCG hesapla
    dcg = dcg_hesapla(ilgi_listesi)
    
    # Ideal DCG hesapla (en iyi sıralama)
    # Tüm ilgi puanlarını büyükten küçüğe sırala
    ideal_ilgi = sorted(gercek_ilgi.values(), reverse=True)[:k]
    idcg = dcg_hesapla(ideal_ilgi)
    
    # NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    
    return float(ndcg)


def tum_metrikleri_hesapla(gercek: list, tahmin: list) -> dict:
    """
    🎯 Bu fonksiyon ne yapar?
    Tüm regresyon metriklerini tek seferde hesaplar.
    
    Bir model değerlendirmesi için gereken tüm
    metrikleri döndürür.
    
    📥 Parametreler:
    - gercek: Gerçek değerler
    - tahmin: Tahmin edilen değerler
    
    📤 Döndürdüğü:
    - metrikler (dict): Tüm metrik değerleri
    
    💡 Örnek Kullanım:
    >>> metrikler = tum_metrikleri_hesapla([5,4,3], [4.5,4,3.5])
    >>> print(metrikler)
    {'RMSE': 0.408, 'MAE': 0.333, 'MSE': 0.167, 'R2': 0.833}
    """
    
    metrikler = {
        'RMSE': rmse_hesapla(gercek, tahmin),
        'MAE': mae_hesapla(gercek, tahmin),
        'MSE': mse_hesapla(gercek, tahmin),
        'R2': r2_hesapla(gercek, tahmin),
    }
    
    # MAPE için sıfır kontrolü yap
    if not any(g == 0 for g in gercek):
        metrikler['MAPE'] = mape_hesapla(gercek, tahmin)
    
    return metrikler


def metrikleri_yazdir(metrikler: dict):
    """
    🎯 Bu fonksiyon ne yapar?
    Metrikleri güzel formatta ekrana yazdırır.
    
    📥 Parametreler:
    - metrikler (dict): Metrik adı → değer eşleşmesi
    """
    
    print("\n" + "=" * 40)
    print("📊 MODEL PERFORMANS METRİKLERİ")
    print("=" * 40)
    
    for metrik_adi, deger in metrikler.items():
        # Özel formatlar
        if metrik_adi == 'R2':
            print(f"   {metrik_adi}: {deger:.4f} ({deger*100:.1f}%)")
        elif metrik_adi == 'MAPE':
            print(f"   {metrik_adi}: %{deger:.2f}")
        else:
            print(f"   {metrik_adi}: {deger:.4f}")
    
    print("=" * 40)
    
    # Yorumlama
    rmse = metrikler.get('RMSE', 0)
    if rmse < 0.5:
        print("✅ Mükemmel! Çok düşük hata.")
    elif rmse < 1.0:
        print("👍 İyi! Kabul edilebilir hata oranı.")
    elif rmse < 1.5:
        print("⚠️ Orta. İyileştirme gerekebilir.")
    else:
        print("❌ Yüksek hata! Model iyileştirilmeli.")


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 Metrik testleri başlıyor...\n")
    
    # Test verileri
    gercek = [5, 4, 3, 5, 4, 3, 2, 5, 4, 3]
    tahmin = [4.5, 4.2, 2.8, 4.8, 3.5, 3.2, 2.5, 4.2, 4.1, 3.0]
    
    # Tüm metrikleri hesapla
    metrikler = tum_metrikleri_hesapla(gercek, tahmin)
    metrikleri_yazdir(metrikler)
    
    # Precision/Recall testi
    print("\n📊 Öneri Metrikleri Testi")
    print("-" * 40)
    onerilen = {1, 5, 10, 15, 20}
    begenilenler = {1, 5, 8, 12, 20}
    
    p = precision_hesapla(onerilen, begenilenler)
    r = recall_hesapla(onerilen, begenilenler)
    f1 = f1_hesapla(p, r)
    
    print(f"   Precision: {p:.2f} (5 önerinin 3'ü doğru)")
    print(f"   Recall: {r:.2f} (5 beğenilenin 3'ü önerildi)")
    print(f"   F1-Score: {f1:.2f}")
    
    print("\n✅ Tüm testler tamamlandı!")
