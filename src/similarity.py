# ============================================
# 📐 Benzerlik Hesaplama Modülü (Similarity)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# İki vektör (kullanıcı veya film) arasındaki
# benzerliği hesaplar.
#
# Benzerlik neden önemli?
# "Sana benzer kullanıcılar şu filmleri sevdi"
# demek için önce benzer kullanıcıları bulmamız lazım!
#
# 💡 Kullanım:
# from src.similarity import kosinus_benzerligi, pearson_korelasyonu
# ============================================

# ----- Gerekli Kütüphaneleri İçe Aktar -----

import numpy as np
from typing import Union, List


# ============================================
# 📐 Kosinüs Benzerliği (Cosine Similarity)
# ============================================
# 
# Formül: cos(θ) = (A · B) / (||A|| × ||B||)
# 
# A · B : Nokta çarpımı (dot product)
# ||A|| : A vektörünün uzunluğu (norm)
# 
# Ne anlama gelir?
# İki vektör arasındaki açıyı ölçer.
# Açı küçükse → Benzerlik yüksek
# Açı büyükse → Benzerlik düşük
# 
# Değer aralığı: -1 ile 1 arası
# - 1: Tamamen aynı yön (çok benzer)
# - 0: Dik açı (ilişkisiz)
# - -1: Zıt yönler (çok farklı)
# 
# Neden kullanılır?
# Vektörlerin büyüklüğünü değil, yönünü karşılaştırır.
# Bir kullanıcı 1-3 arası, başkası 3-5 arası puan verse bile
# aynı tercihlere sahipse benzer sayılır!
# 
# Gerçek hayat örneği:
# Kullanıcı A: [5, 4, 0, 1] (Aksiyon sever, Romantik sevmez)
# Kullanıcı B: [4, 5, 1, 0] (Aksiyon sever, Romantik sevmez)
# Kosinüs benzerliği yüksek olacak!
# ============================================

def kosinus_benzerligi(vektor1: Union[np.ndarray, List], 
                       vektor2: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    İki vektör arasındaki kosinüs benzerliğini hesaplar.
    
    Düşün ki iki ok (vektör) var ve aralarındaki
    açıyı ölçüyorsun. Açı küçükse oklar benzer
    yöne bakıyor demektir!
    
    📥 Parametreler:
    - vektor1: İlk vektör (sayı listesi)
      Örnek: [5, 4, 0, 1] (kullanıcının film puanları)
    - vektor2: İkinci vektör
      Örnek: [4, 5, 1, 0] (başka kullanıcının puanları)
    
    📤 Döndürdüğü:
    - benzerlik (float): -1 ile 1 arası
      1'e yakın = çok benzer
      0'a yakın = ilişkisiz
      -1'e yakın = zıt
    
    💡 Örnek Kullanım:
    >>> v1 = [5, 4, 3, 2, 1]
    >>> v2 = [4, 5, 2, 3, 1]
    >>> benzerlik = kosinus_benzerligi(v1, v2)
    >>> print(f"Benzerlik: {benzerlik:.4f}")
    Benzerlik: 0.9636
    
    🤔 Neden bu fonksiyonu yazdık?
    İşbirlikçi filtrelemede benzer kullanıcıları
    veya benzer filmleri bulmak için!
    """
    
    # Numpy dizisine çevir
    v1 = np.array(vektor1, dtype=float)
    v2 = np.array(vektor2, dtype=float)
    
    # Boyut kontrolü
    if len(v1) != len(v2):
        raise ValueError(
            f"❌ Vektör boyutları eşit olmalı! "
            f"v1: {len(v1)}, v2: {len(v2)}"
        )
    
    # ----- Adım 1: Nokta Çarpımı (Dot Product) -----
    # Her elemanı birbiriyle çarp ve topla
    # Örnek: [1,2] · [3,4] = 1*3 + 2*4 = 11
    nokta_carpimi = np.dot(v1, v2)
    
    # ----- Adım 2: Vektör Uzunlukları (Norms) -----
    # Pisagor teoremi gibi: √(x² + y² + z²)
    # Örnek: ||[3,4]|| = √(9+16) = 5
    uzunluk1 = np.linalg.norm(v1)
    uzunluk2 = np.linalg.norm(v2)
    
    # ----- Adım 3: Sıfır Kontrolü -----
    # Uzunluk 0 ise (tüm elemanlar 0), benzerlik tanımsız
    if uzunluk1 == 0 or uzunluk2 == 0:
        return 0.0
    
    # ----- Adım 4: Kosinüs Benzerliği -----
    # Formül: nokta_çarpımı / (uzunluk1 × uzunluk2)
    benzerlik = nokta_carpimi / (uzunluk1 * uzunluk2)
    
    return float(benzerlik)


# ============================================
# 📐 Pearson Korelasyonu (Pearson Correlation)
# ============================================
# 
# Formül: r = Σ(x-x̄)(y-ȳ) / √(Σ(x-x̄)² × Σ(y-ȳ)²)
# 
# x̄ : x'in ortalaması
# ȳ : y'nin ortalaması
# 
# Kosinüs benzerliğinden farkı:
# Pearson, ortalamadan sapmaları ölçer.
# "Her ikisi de ortalamanın üstünde mi, altında mı?"
# 
# Değer aralığı: -1 ile 1 arası
# - 1: Mükemmel pozitif ilişki
# - 0: İlişki yok
# - -1: Mükemmel negatif ilişki
# 
# Ne zaman kullanılır?
# Kullanıcıların puan verme alışkanlıkları farklıysa.
# Biri hep 4-5, başkası hep 2-3 veriyorsa,
# Pearson bu farkı normalize eder!
# 
# Gerçek hayat örneği:
# Kullanıcı A: [5, 5, 4, 5] - Hep yüksek puan verir
# Kullanıcı B: [3, 3, 2, 3] - Hep düşük puan verir
# Ama ikisi de aynı filmleri seviyor! Pearson bunu yakalar.
# ============================================

def pearson_korelasyonu(vektor1: Union[np.ndarray, List], 
                        vektor2: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    İki vektör arasındaki Pearson korelasyonunu hesaplar.
    
    Her iki vektörü de kendi ortalamalarından çıkarır,
    sonra kosinüs benzerliği gibi karşılaştırır.
    
    Bu sayede puan verme eğilimlerindeki farkları
    etkisiz hale getirir!
    
    📥 Parametreler:
    - vektor1: İlk vektör (sayı listesi)
    - vektor2: İkinci vektör
    
    📤 Döndürdüğü:
    - korelasyon (float): -1 ile 1 arası
    
    💡 Örnek Kullanım:
    >>> v1 = [5, 5, 4, 5, 4]  # Yüksek puan veren kullanıcı
    >>> v2 = [3, 3, 2, 3, 2]  # Düşük puan veren kullanıcı
    >>> r = pearson_korelasyonu(v1, v2)
    >>> print(f"Korelasyon: {r:.4f}")
    Korelasyon: 1.0000  # Aynı tercihlere sahipler!
    
    🤔 Neden bu fonksiyonu yazdık?
    Bazı kullanıcılar cimri (hep düşük puan),
    bazıları cömert (hep yüksek puan).
    Pearson bu farkı düzeltir ve gerçek tercihleri bulur.
    """
    
    v1 = np.array(vektor1, dtype=float)
    v2 = np.array(vektor2, dtype=float)
    
    if len(v1) != len(v2):
        raise ValueError(f"❌ Vektör boyutları eşit olmalı!")
    
    # ----- Adım 1: Ortalamaları Hesapla -----
    ortalama1 = np.mean(v1)
    ortalama2 = np.mean(v2)
    
    # ----- Adım 2: Ortalamaları Çıkar (Merkezleme) -----
    # Her elemanı kendi ortalamasından çıkar
    # Bu işleme "centering" denir
    v1_merkezli = v1 - ortalama1
    v2_merkezli = v2 - ortalama2
    
    # ----- Adım 3: Payı Hesapla -----
    # Σ(x-x̄)(y-ȳ) = v1_merkezli · v2_merkezli
    pay = np.sum(v1_merkezli * v2_merkezli)
    
    # ----- Adım 4: Paydayı Hesapla -----
    # √(Σ(x-x̄)² × Σ(y-ȳ)²)
    kare_toplam1 = np.sum(v1_merkezli ** 2)
    kare_toplam2 = np.sum(v2_merkezli ** 2)
    payda = np.sqrt(kare_toplam1 * kare_toplam2)
    
    # ----- Adım 5: Sıfır Kontrolü -----
    if payda == 0:
        return 0.0
    
    # ----- Adım 6: Korelasyonu Hesapla -----
    korelasyon = pay / payda
    
    return float(korelasyon)


# ============================================
# 📐 Öklid Uzaklığı (Euclidean Distance)
# ============================================
# 
# Formül: d = √(Σ(x_i - y_i)²)
# 
# İki nokta arasındaki "kuş uçuşu" mesafe.
# Pisagor teoreminin çok boyutlu hali!
# 
# 2D örnek: (0,0) ile (3,4) arası
# d = √(3² + 4²) = √(9+16) = √25 = 5
# 
# Dikkat: Bu bir uzaklık, benzerlik değil!
# Uzaklık küçükse → Benzerlik yüksek
# Uzaklık büyükse → Benzerlik düşük
# ============================================

def oklid_uzakligi(vektor1: Union[np.ndarray, List], 
                   vektor2: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    İki vektör arasındaki Öklid uzaklığını hesaplar.
    
    Düşün ki iki noktan var ve arasındaki
    düz çizgi mesafesini ölçüyorsun.
    
    📥 Parametreler:
    - vektor1: İlk vektör
    - vektor2: İkinci vektör
    
    📤 Döndürdüğü:
    - uzaklik (float): 0 ile sonsuz arası
      0 = Aynı nokta
      Küçük değer = Yakın (benzer)
      Büyük değer = Uzak (farklı)
    
    💡 Örnek Kullanım:
    >>> v1 = [0, 0]
    >>> v2 = [3, 4]
    >>> uzaklik = oklid_uzakligi(v1, v2)
    >>> print(f"Uzaklık: {uzaklik}")
    Uzaklık: 5.0
    
    🤔 Neden bu fonksiyonu yazdık?
    KNN (K-Nearest Neighbors) gibi algoritmalar
    uzaklık temelli çalışır. En yakın komşuları
    bulmak için uzaklık hesaplarız.
    """
    
    v1 = np.array(vektor1, dtype=float)
    v2 = np.array(vektor2, dtype=float)
    
    if len(v1) != len(v2):
        raise ValueError(f"❌ Vektör boyutları eşit olmalı!")
    
    # Farkların karelerinin toplamının karekökü
    # numpy.linalg.norm ile de hesaplanabilir
    uzaklik = np.sqrt(np.sum((v1 - v2) ** 2))
    
    return float(uzaklik)


def oklid_benzerligi(vektor1: Union[np.ndarray, List], 
                     vektor2: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    Öklid uzaklığını benzerliğe çevirir.
    
    Formül: benzerlik = 1 / (1 + uzaklık)
    
    Böylece:
    - Uzaklık = 0 → Benzerlik = 1
    - Uzaklık = ∞ → Benzerlik = 0
    
    📥 Parametreler:
    - vektor1, vektor2: Karşılaştırılacak vektörler
    
    📤 Döndürdüğü:
    - benzerlik (float): 0 ile 1 arası
    """
    
    uzaklik = oklid_uzakligi(vektor1, vektor2)
    
    # 1 + uzaklık ile böleriz
    # +1 sayesinde 0'a bölme olmaz
    benzerlik = 1 / (1 + uzaklik)
    
    return float(benzerlik)


# ============================================
# 📐 Manhattan Uzaklığı (Manhattan Distance)
# ============================================
# 
# Formül: d = Σ|x_i - y_i|
# 
# Şehir bloklarında yürür gibi!
# New York'ta A noktasından B'ye giderken
# köşegenlerden değil, sokaklardan yürürsün.
# 
# Diğer adları: L1 mesafesi, Taksi mesafesi
# 
# 2D örnek: (0,0) ile (3,4) arası
# Manhattan = |3-0| + |4-0| = 7
# (Öklid = 5 idi)
# ============================================

def manhattan_uzakligi(vektor1: Union[np.ndarray, List], 
                       vektor2: Union[np.ndarray, List]) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    İki vektör arasındaki Manhattan uzaklığını hesaplar.
    
    Düşün ki New York'tasın ve sokaklar ızgara şeklinde.
    Köşegenlerden gidemezsin, sokaklardan yürümelisin!
    
    📥 Parametreler:
    - vektor1, vektor2: Karşılaştırılacak vektörler
    
    📤 Döndürdüğü:
    - uzaklik (float): 0 ile sonsuz arası
    
    💡 Örnek Kullanım:
    >>> manhattan_uzakligi([0, 0], [3, 4])
    7.0
    """
    
    v1 = np.array(vektor1, dtype=float)
    v2 = np.array(vektor2, dtype=float)
    
    if len(v1) != len(v2):
        raise ValueError(f"❌ Vektör boyutları eşit olmalı!")
    
    # Mutlak farkların toplamı
    uzaklik = np.sum(np.abs(v1 - v2))
    
    return float(uzaklik)


# ============================================
# 📐 Jaccard Benzerliği (Jaccard Similarity)
# ============================================
# 
# Formül: J(A,B) = |A ∩ B| / |A ∪ B|
# 
# ∩ = Kesişim (her ikisinde de olan)
# ∪ = Birleşim (en az birinde olan)
# 
# Küme karşılaştırması için kullanılır.
# "İki kullanıcı kaç ortak film izlemiş?"
# 
# Değer aralığı: 0 ile 1 arası
# 0 = Hiç ortak yok
# 1 = Tamamen aynı
# ============================================

def jaccard_benzerligi(kume1: set, kume2: set) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    İki küme arasındaki Jaccard benzerliğini hesaplar.
    
    "İki kullanıcının izlediği filmler ne kadar örtüşüyor?"
    sorusuna cevap verir.
    
    📥 Parametreler:
    - kume1 (set): İlk küme
      Örnek: {1, 2, 3, 4, 5} (kullanıcı 1'in izlediği filmler)
    - kume2 (set): İkinci küme
      Örnek: {3, 4, 5, 6, 7} (kullanıcı 2'nin izlediği filmler)
    
    📤 Döndürdüğü:
    - benzerlik (float): 0 ile 1 arası
    
    💡 Örnek Kullanım:
    >>> k1 = {1, 2, 3, 4, 5}
    >>> k2 = {3, 4, 5, 6, 7}
    >>> j = jaccard_benzerligi(k1, k2)
    >>> print(f"Jaccard: {j:.2f}")
    Jaccard: 0.43  # 3 ortak / 7 toplam = 3/7
    
    🤔 Neden bu fonksiyonu yazdık?
    Sadece izlenip izlenmediğini önemsediğimizde kullanılır.
    Puan bilgisi yoksa çok faydalı!
    """
    
    # Set'e çevir (liste olarak gelebilir)
    kume1 = set(kume1)
    kume2 = set(kume2)
    
    # Kesişim: Her ikisinde de olan
    kesisim = kume1 & kume2
    
    # Birleşim: En az birinde olan
    birlesim = kume1 | kume2
    
    # Birleşim boşsa, 0 döndür
    if len(birlesim) == 0:
        return 0.0
    
    # Jaccard = Kesişim / Birleşim
    benzerlik = len(kesisim) / len(birlesim)
    
    return float(benzerlik)


# ============================================
# 📐 Adjusted Cosine Similarity
# Düzeltilmiş Kosinüs Benzerliği
# ============================================
# 
# Öğe-tabanlı işbirlikçi filtreleme için özel!
# 
# Normal kosinüste problem:
# Bir kullanıcı hep 5 veriyorsa ve başkası hep 3 veriyorsa,
# aynı filmi beğenseler bile farklı puanlar verirler.
# 
# Çözüm:
# Her kullanıcının puanlarından kendi ortalamasını çıkar.
# Sonra kosinüs hesapla.
# ============================================

def adjusted_kosinus_benzerligi(matris: np.ndarray, 
                                 item1_idx: int, 
                                 item2_idx: int) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    İki öğe (film) arasındaki düzeltilmiş kosinüs benzerliğini hesaplar.
    
    Her kullanıcının puan verme eğilimini normalize eder.
    Böylece cimri ve cömert kullanıcılar eşit sayılır!
    
    📥 Parametreler:
    - matris (np.ndarray): Kullanıcı-Film puan matrisi
      Satırlar: Kullanıcılar
      Sütunlar: Filmler
    - item1_idx (int): İlk filmin sütun indeksi
    - item2_idx (int): İkinci filmin sütun indeksi
    
    📤 Döndürdüğü:
    - benzerlik (float): -1 ile 1 arası
    
    🤔 Neden bu fonksiyonu yazdık?
    Öğe-tabanlı işbirlikçi filtreleme için standart yöntem!
    "Bu filmi sevenler, şu filmi de seviyor" bulmak için.
    """
    
    # İlgili sütunları (filmleri) al
    item1 = matris[:, item1_idx]
    item2 = matris[:, item2_idx]
    
    # Her iki filmi de puanlayan kullanıcıları bul
    # ~np.isnan: NaN olmayan (yani puan verilmiş) yerler
    ortak_maske = ~np.isnan(item1) & ~np.isnan(item2)
    
    # Yeterli ortak puan yoksa, 0 döndür
    if ortak_maske.sum() < 2:
        return 0.0
    
    # Ortak kullanıcıların puanlarını al
    item1_ortak = item1[ortak_maske]
    item2_ortak = item2[ortak_maske]
    
    # Her kullanıcının ortalamasını hesapla ve çıkar
    kullanici_ortalamalari = np.nanmean(matris, axis=1)
    ortalamalar = kullanici_ortalamalari[ortak_maske]
    
    # Ortalamaları çıkar (normalize et)
    item1_norm = item1_ortak - ortalamalar
    item2_norm = item2_ortak - ortalamalar
    
    # Kosinüs hesapla
    pay = np.sum(item1_norm * item2_norm)
    payda = np.sqrt(np.sum(item1_norm ** 2)) * np.sqrt(np.sum(item2_norm ** 2))
    
    if payda == 0:
        return 0.0
    
    benzerlik = pay / payda
    
    return float(benzerlik)


def benzerlik_matrisi_olustur(matris: np.ndarray, 
                               yontem: str = "kosinus",
                               eksen: str = "kullanici") -> np.ndarray:
    """
    🎯 Bu fonksiyon ne yapar?
    Tüm kullanıcılar veya filmler arası benzerlik matrisini oluşturur.
    
    📥 Parametreler:
    - matris: Kullanıcı-Film puan matrisi
    - yontem: "kosinus" veya "pearson"
    - eksen: "kullanici" veya "film"
    
    📤 Döndürdüğü:
    - benzerlik_matrisi (np.ndarray): Benzerlik değerleri
    
    💡 Örnek Kullanım:
    >>> benz_matris = benzerlik_matrisi_olustur(puan_matrisi, "kosinus", "kullanici")
    >>> print(benz_matris[0, 1])  # Kullanıcı 0 ve 1 arası benzerlik
    """
    
    # Film bazlı hesaplama için matrisi transpoze et
    if eksen == "film":
        matris = matris.T
    
    n = matris.shape[0]
    benzerlik_matrisi = np.zeros((n, n))
    
    # Tüm çiftler için benzerlik hesapla
    for i in range(n):
        for j in range(i, n):
            if i == j:
                benzerlik_matrisi[i, j] = 1.0  # Kendisiyle benzerlik = 1
            else:
                # NaN değerlerini 0 ile değiştir
                v1 = np.nan_to_num(matris[i])
                v2 = np.nan_to_num(matris[j])
                
                if yontem == "kosinus":
                    benzerlik = kosinus_benzerligi(v1, v2)
                elif yontem == "pearson":
                    benzerlik = pearson_korelasyonu(v1, v2)
                else:
                    raise ValueError(f"Bilinmeyen yöntem: {yontem}")
                
                # Simetrik matris
                benzerlik_matrisi[i, j] = benzerlik
                benzerlik_matrisi[j, i] = benzerlik
    
    return benzerlik_matrisi


def en_benzer_bul(benzerlik_matrisi: np.ndarray, 
                  hedef_idx: int, 
                  k: int = 10) -> List[tuple]:
    """
    🎯 Bu fonksiyon ne yapar?
    En benzer k komşuyu bulur.
    
    📥 Parametreler:
    - benzerlik_matrisi: Tüm benzerlik değerleri
    - hedef_idx: Hedef kullanıcı/film indeksi
    - k: Kaç benzer bulunacak
    
    📤 Döndürdüğü:
    - benzerler (list): [(indeks, benzerlik), ...] listesi
    
    💡 Örnek Kullanım:
    >>> benzerler = en_benzer_bul(benz_matris, hedef_idx=5, k=3)
    >>> print(benzerler)
    [(12, 0.95), (7, 0.89), (23, 0.85)]
    """
    
    # Hedefin benzerlik satırını al
    benzerlikler = benzerlik_matrisi[hedef_idx]
    
    # Kendisi hariç en yüksek k değeri bul
    # argsort: Küçükten büyüğe sıralar, [::-1] ile tersine çevir
    sirali_indeksler = np.argsort(benzerlikler)[::-1]
    
    # Kendisini çıkar (benzerlik = 1.0 olanı atla)
    benzerler = []
    for idx in sirali_indeksler:
        if idx != hedef_idx and len(benzerler) < k:
            benzerler.append((idx, benzerlikler[idx]))
    
    return benzerler


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 Benzerlik testleri başlıyor...\n")
    
    # Test vektörleri
    v1 = [5, 4, 3, 2, 1]
    v2 = [4, 5, 2, 3, 1]
    v3 = [1, 2, 3, 4, 5]  # v1'in tersi
    
    print("Test vektörleri:")
    print(f"  v1 = {v1}")
    print(f"  v2 = {v2}")
    print(f"  v3 = {v3} (v1'in tersi)")
    
    print("\n" + "-" * 40)
    print("📐 Kosinüs Benzerliği:")
    print(f"  v1-v2: {kosinus_benzerligi(v1, v2):.4f} (benzer)")
    print(f"  v1-v3: {kosinus_benzerligi(v1, v3):.4f} (farklı)")
    
    print("\n📐 Pearson Korelasyonu:")
    print(f"  v1-v2: {pearson_korelasyonu(v1, v2):.4f}")
    print(f"  v1-v3: {pearson_korelasyonu(v1, v3):.4f} (negatif!)")
    
    print("\n📐 Öklid Uzaklığı:")
    print(f"  v1-v2: {oklid_uzakligi(v1, v2):.4f}")
    print(f"  v1-v3: {oklid_uzakligi(v1, v3):.4f}")
    
    # Jaccard testi
    k1 = {1, 2, 3, 4, 5}
    k2 = {3, 4, 5, 6, 7}
    print("\n📐 Jaccard Benzerliği:")
    print(f"  Küme1: {k1}")
    print(f"  Küme2: {k2}")
    print(f"  Jaccard: {jaccard_benzerligi(k1, k2):.4f}")
    
    print("\n✅ Tüm testler tamamlandı!")
