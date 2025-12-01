# ============================================
# 🔧 Veri Ön İşleme Modülü (Preprocessing)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Veriyi temizler ve modellerin kullanabileceği
# hale getirir. Mutfakta yemek yapmadan önce
# malzemeleri yıkayıp doğramak gibi!
#
# 💡 Kullanım:
# from src.preprocessing import veri_temizle, normalize_et
# ============================================

# ----- Gerekli Kütüphaneleri İçe Aktar -----

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def eksik_verileri_temizle(df: pd.DataFrame, strateji: str = "sil") -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Tablodaki eksik (boş) verileri temizler.
    
    Düşün ki bir sınav kağıdında bazı sorular boş.
    Bu fonksiyon o boş soruları ya siliyor,
    ya da ortalama ile dolduruyor.
    
    📥 Parametreler:
    - df (DataFrame): Temizlenecek tablo
    - strateji (str): Temizleme yöntemi
      - "sil": Eksik satırları sil
      - "ortalama": Sayısal değerleri ortalama ile doldur
      - "medyan": Sayısal değerleri medyan ile doldur
      - "sifir": Sıfır ile doldur
    
    📤 Döndürdüğü:
    - temiz_df (DataFrame): Temizlenmiş tablo
    
    💡 Örnek Kullanım:
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> temiz_df = eksik_verileri_temizle(df, strateji="ortalama")
    
    🤔 Neden bu fonksiyonu yazdık?
    Makine öğrenmesi modelleri eksik verilerle çalışamaz!
    Modeli eğitmeden önce veriyi temizlememiz lazım.
    """
    
    # Başlangıçtaki eksik veri sayısını bul
    eksik_sayisi = df.isnull().sum().sum()
    
    if eksik_sayisi == 0:
        print("✅ Eksik veri yok, temizleme gerekmiyor!")
        return df
    
    print(f"🔍 Toplam {eksik_sayisi:,} eksik veri bulundu")
    
    # Kopyasını al (orijinali değiştirmemek için)
    temiz_df = df.copy()
    
    if strateji == "sil":
        # Eksik veri içeren satırları sil
        # dropna: NA (boş) değerleri düşür
        temiz_df = temiz_df.dropna()
        print(f"   ❌ {len(df) - len(temiz_df):,} satır silindi")
        
    elif strateji == "ortalama":
        # Sayısal sütunları ortalama ile doldur
        sayisal_sutunlar = temiz_df.select_dtypes(include=[np.number]).columns
        for sutun in sayisal_sutunlar:
            ortalama = temiz_df[sutun].mean()
            temiz_df[sutun].fillna(ortalama, inplace=True)
        print(f"   📊 Sayısal değerler ortalama ile dolduruldu")
        
    elif strateji == "medyan":
        # Sayısal sütunları medyan ile doldur
        # Medyan: Ortadaki değer (uç değerlerden etkilenmez)
        sayisal_sutunlar = temiz_df.select_dtypes(include=[np.number]).columns
        for sutun in sayisal_sutunlar:
            medyan = temiz_df[sutun].median()
            temiz_df[sutun].fillna(medyan, inplace=True)
        print(f"   📊 Sayısal değerler medyan ile dolduruldu")
        
    elif strateji == "sifir":
        # Tüm eksik değerleri sıfır ile doldur
        temiz_df = temiz_df.fillna(0)
        print(f"   0️⃣ Eksik değerler sıfır ile dolduruldu")
        
    else:
        raise ValueError(f"❌ Bilinmeyen strateji: {strateji}")
    
    return temiz_df


def puan_normalize_et(puanlar: pd.Series, yontem: str = "min-max") -> pd.Series:
    """
    🎯 Bu fonksiyon ne yapar?
    Puanları belirli bir aralığa normalize eder (standartlaştırır).
    
    Normalizasyon nedir?
    Farklı ölçeklerdeki değerleri aynı ölçeğe getirmek.
    
    Örnek:
    - Bir sitede puanlar 1-10 arası
    - Başka sitede 1-5 arası
    - Normalize edince ikisi de 0-1 arası olur!
    
    📥 Parametreler:
    - puanlar (Series): Normalize edilecek puanlar
    - yontem (str): Normalizasyon yöntemi
      - "min-max": 0-1 arasına normalize (varsayılan)
      - "z-score": Ortalama=0, Standart sapma=1
    
    📤 Döndürdüğü:
    - normalize_puanlar (Series): Normalize edilmiş puanlar
    
    💡 Örnek Kullanım:
    >>> puanlar = pd.Series([1, 2, 3, 4, 5])
    >>> norm_puanlar = puan_normalize_et(puanlar, yontem="min-max")
    >>> print(norm_puanlar)
    0    0.00
    1    0.25
    2    0.50
    3    0.75
    4    1.00
    
    🤔 Neden bu fonksiyonu yazdık?
    Bazı algoritmalar normalize edilmiş verilerle
    daha iyi çalışır. Özellikle uzaklık tabanlı
    algoritmalar için önemli!
    """
    
    if yontem == "min-max":
        # Min-Max Normalizasyon
        # Formül: (x - min) / (max - min)
        # Sonuç: 0 ile 1 arasında
        
        min_deger = puanlar.min()  # En küçük puan
        max_deger = puanlar.max()  # En büyük puan
        aralik = max_deger - min_deger  # Puan aralığı
        
        # Aralık 0 ise (tüm puanlar aynı), sıfır döndür
        if aralik == 0:
            return pd.Series([0.5] * len(puanlar))
        
        # Normalizasyon formülünü uygula
        normalize_puanlar = (puanlar - min_deger) / aralik
        
    elif yontem == "z-score":
        # Z-Score (Standart) Normalizasyon
        # Formül: (x - ortalama) / standart_sapma
        # Sonuç: Ortalama=0, Std=1
        
        ortalama = puanlar.mean()  # Ortalama puan
        std = puanlar.std()  # Standart sapma
        
        if std == 0:
            return pd.Series([0] * len(puanlar))
        
        normalize_puanlar = (puanlar - ortalama) / std
        
    else:
        raise ValueError(f"❌ Bilinmeyen yöntem: {yontem}")
    
    return normalize_puanlar


def egitim_test_ayir(
    puanlar: pd.DataFrame,
    test_orani: float = 0.2,
    rastgele_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    🎯 Bu fonksiyon ne yapar?
    Veriyi eğitim ve test setlerine ayırır.
    
    Düşün ki bir sınava hazırlanıyorsun:
    - Eğitim seti: Çalıştığın konular (modeli eğitmek için)
    - Test seti: Sınav soruları (modeli test etmek için)
    
    Neden ayırıyoruz?
    Modelin "kopya çekmesini" engellemek için!
    Eğitimde görmediği veriyle test ediyoruz.
    
    📥 Parametreler:
    - puanlar (DataFrame): Tüm puan verileri
    - test_orani (float): Test setinin oranı (0-1 arası)
      Örnek: 0.2 = %20 test, %80 eğitim
    - rastgele_seed (int): Rastgelelik için tohum değeri
      Aynı seed = Aynı bölme sonucu (tekrarlanabilirlik)
    
    📤 Döndürdüğü:
    - egitim_seti (DataFrame): Eğitim verileri
    - test_seti (DataFrame): Test verileri
    
    💡 Örnek Kullanım:
    >>> filmler, puanlar = veri_yukle()
    >>> egitim, test = egitim_test_ayir(puanlar, test_orani=0.2)
    >>> print(f"Eğitim: {len(egitim)}, Test: {len(test)}")
    Eğitim: 80000, Test: 20000
    
    🤔 Neden bu fonksiyonu yazdık?
    Model performansını doğru ölçmek için!
    Eğitimde kullanılan veriyle test etmek = Kendini kandırmak!
    """
    
    # train_test_split: sklearn'ün veri bölme fonksiyonu
    egitim_seti, test_seti = train_test_split(
        puanlar,
        test_size=test_orani,  # Test oranı
        random_state=rastgele_seed,  # Tekrarlanabilirlik için
        shuffle=True  # Veriyi karıştır
    )
    
    # Bilgi mesajı
    toplam = len(puanlar)
    egitim_adet = len(egitim_seti)
    test_adet = len(test_seti)
    
    print("✅ Veri bölme tamamlandı!")
    print(f"   📚 Eğitim seti: {egitim_adet:,} ({egitim_adet/toplam*100:.1f}%)")
    print(f"   🧪 Test seti: {test_adet:,} ({test_adet/toplam*100:.1f}%)")
    
    return egitim_seti, test_seti


def kullanici_bazli_ayir(
    puanlar: pd.DataFrame,
    test_orani: float = 0.2,
    rastgele_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    🎯 Bu fonksiyon ne yapar?
    Her kullanıcının puanlarının bir kısmını teste ayırır.
    
    Normal train_test_split'ten farkı:
    - Normal: Rastgele satırları ayırır
    - Bu fonksiyon: Her kullanıcıdan eşit oranda ayırır
    
    Bu önemli çünkü:
    Bir kullanıcının TÜM verileri testte olursa,
    eğitimde o kullanıcıyı hiç görmemiş oluruz!
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu
    - test_orani (float): Her kullanıcıdan ayrılacak oran
    - rastgele_seed (int): Rastgelelik tohumu
    
    📤 Döndürdüğü:
    - egitim_seti, test_seti (DataFrame): Ayrılmış veriler
    """
    
    # Rastgelelik için seed ayarla
    np.random.seed(rastgele_seed)
    
    egitim_listesi = []  # Eğitim verilerini tutacak
    test_listesi = []     # Test verilerini tutacak
    
    # Her kullanıcı için ayrı ayrı işlem yap
    for kullanici_id in puanlar['userId'].unique():
        # Bu kullanıcının tüm puanlarını al
        kullanici_puanlari = puanlar[puanlar['userId'] == kullanici_id]
        
        # Eğer yeterli puan varsa, böl
        if len(kullanici_puanlari) >= 2:
            # Rastgele karıştır ve böl
            egitim, test = train_test_split(
                kullanici_puanlari,
                test_size=test_orani,
                random_state=rastgele_seed
            )
            egitim_listesi.append(egitim)
            test_listesi.append(test)
        else:
            # Tek puan varsa, eğitime ekle
            egitim_listesi.append(kullanici_puanlari)
    
    # Listeleri birleştir
    egitim_seti = pd.concat(egitim_listesi, ignore_index=True)
    test_seti = pd.concat(test_listesi, ignore_index=True)
    
    print("✅ Kullanıcı bazlı bölme tamamlandı!")
    print(f"   📚 Eğitim: {len(egitim_seti):,}")
    print(f"   🧪 Test: {len(test_seti):,}")
    
    return egitim_seti, test_seti


def film_turlerini_ayir(filmler: pd.DataFrame) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Film türlerini ayrı sütunlara böler (One-Hot Encoding).
    
    Giriş:
    movieId | title      | genres
    1       | Toy Story  | Animation|Children|Comedy
    
    Çıkış:
    movieId | title      | Animation | Children | Comedy | ...
    1       | Toy Story  | 1         | 1        | 1      | ...
    
    📥 Parametreler:
    - filmler (DataFrame): Film tablosu (genres sütunu olmalı)
    
    📤 Döndürdüğü:
    - genisletilmis_filmler (DataFrame): Türler ayrı sütunlarda
    
    🤔 Neden bu fonksiyonu yazdık?
    İçerik tabanlı filtreleme için film özelliklerini
    sayısal formata çevirmemiz gerekiyor!
    """
    
    # Kopyasını al
    df = filmler.copy()
    
    # Tüm benzersiz türleri bul
    tum_turler = set()
    for turler in df['genres']:
        # Her filmin türlerini | ile ayır
        if turler != "(no genres listed)":
            tum_turler.update(turler.split('|'))
    
    print(f"🎭 {len(tum_turler)} farklı tür bulundu: {sorted(tum_turler)}")
    
    # Her tür için yeni sütun oluştur
    for tur in tum_turler:
        # lambda: Tek satırlık fonksiyon
        # Film o türe aitse 1, değilse 0
        df[tur] = df['genres'].apply(
            lambda x: 1 if tur in x.split('|') else 0
        )
    
    print("✅ Tür sütunları oluşturuldu!")
    
    return df


def tarih_ozelliklerini_cikar(puanlar: pd.DataFrame) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Timestamp (zaman damgası) sütunundan tarih özellikleri çıkarır.
    
    Timestamp: 964982703 gibi bir sayı
    Bu aslında: 2000-07-30 18:45:03 tarihini temsil eder
    (Unix timestamp - 1 Ocak 1970'den bu yana geçen saniye)
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu (timestamp sütunu olmalı)
    
    📤 Döndürdüğü:
    - puanlar (DataFrame): Tarih özellikleri eklenmiş tablo
    
    🤔 Neden bu fonksiyonu yazdık?
    Kullanıcıların hangi saatte, günde veya ayda
    film izlediğini analiz edebiliriz. Belki
    hafta sonu daha çok film izleniyordur?
    """
    
    df = puanlar.copy()
    
    # Timestamp'ı datetime'a çevir
    # unit='s': Saniye cinsinden
    df['tarih'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Yıl
    df['yil'] = df['tarih'].dt.year
    
    # Ay (1-12)
    df['ay'] = df['tarih'].dt.month
    
    # Gün (1-31)
    df['gun'] = df['tarih'].dt.day
    
    # Haftanın günü (0=Pazartesi, 6=Pazar)
    df['hafta_gunu'] = df['tarih'].dt.dayofweek
    
    # Saat (0-23)
    df['saat'] = df['tarih'].dt.hour
    
    # Hafta sonu mu? (Cumartesi veya Pazar)
    df['hafta_sonu'] = df['hafta_gunu'].isin([5, 6]).astype(int)
    
    print("✅ Tarih özellikleri çıkarıldı!")
    print(f"   📅 Yıl aralığı: {df['yil'].min()} - {df['yil'].max()}")
    
    return df


def seyrek_kullanicilari_filtrele(
    puanlar: pd.DataFrame,
    min_puan: int = 5
) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Çok az puan veren kullanıcıları filtreler.
    
    Neden filtreleriz?
    Sadece 1-2 film puanlayan kullanıcıdan
    anlamlı öneri üretmek zor. Daha fazla veriye ihtiyaç var!
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu
    - min_puan (int): Minimum puan sayısı
      Örnek: 5 = En az 5 film puanlamamış kullanıcıları sil
    
    📤 Döndürdüğü:
    - filtrelenmis_puanlar (DataFrame): Filtrelenmiş tablo
    """
    
    # Her kullanıcının kaç puan verdiğini say
    puan_sayilari = puanlar.groupby('userId').size()
    
    # Yeterli puan veren kullanıcıları bul
    yeterli_kullanicilar = puan_sayilari[puan_sayilari >= min_puan].index
    
    # Bu kullanıcıların puanlarını filtrele
    filtrelenmis = puanlar[puanlar['userId'].isin(yeterli_kullanicilar)]
    
    silinen = puanlar['userId'].nunique() - len(yeterli_kullanicilar)
    print(f"✅ {silinen:,} kullanıcı filtrelendi (< {min_puan} puan)")
    print(f"   Kalan kullanıcı: {len(yeterli_kullanicilar):,}")
    
    return filtrelenmis


def seyrek_filmleri_filtrele(
    puanlar: pd.DataFrame,
    min_puan: int = 5
) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Çok az puanlanan filmleri filtreler.
    
    Neden filtreleriz?
    Sadece 1-2 kişinin izlediği filmler için
    benzerlik hesaplamak güvenilir değil.
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu
    - min_puan (int): Minimum puan sayısı
    
    📤 Döndürdüğü:
    - filtrelenmis_puanlar (DataFrame): Filtrelenmiş tablo
    """
    
    # Her filmin kaç puan aldığını say
    puan_sayilari = puanlar.groupby('movieId').size()
    
    # Yeterli puan alan filmleri bul
    yeterli_filmler = puan_sayilari[puan_sayilari >= min_puan].index
    
    # Bu filmlerin puanlarını filtrele
    filtrelenmis = puanlar[puanlar['movieId'].isin(yeterli_filmler)]
    
    silinen = puanlar['movieId'].nunique() - len(yeterli_filmler)
    print(f"✅ {silinen:,} film filtrelendi (< {min_puan} puan)")
    print(f"   Kalan film: {len(yeterli_filmler):,}")
    
    return filtrelenmis


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 Ön işleme modülü testi\n")
    
    # Test verisi oluştur
    test_df = pd.DataFrame({
        'A': [1, 2, None, 4, 5],
        'B': [None, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, 5]
    })
    
    print("Orijinal veri:")
    print(test_df)
    print()
    
    # Eksik veri temizleme testi
    temiz = eksik_verileri_temizle(test_df.copy(), strateji="ortalama")
    print("\nOrtalama ile doldurulmuş:")
    print(temiz)
    
    # Normalizasyon testi
    puanlar = pd.Series([1, 2, 3, 4, 5])
    norm = puan_normalize_et(puanlar, yontem="min-max")
    print("\nNormalize edilmiş puanlar:")
    print(norm)
    
    print("\n✅ Testler tamamlandı!")
