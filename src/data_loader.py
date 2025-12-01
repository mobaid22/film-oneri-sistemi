# ============================================
# 📂 Veri Yükleme Modülü (Data Loader)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# MovieLens veri setini okur ve kullanıma hazır hale getirir.
# Düşün ki bir kütüphaneden kitap alıyorsun,
# bu modül o kitapları alıp önüne koyuyor!
#
# 💡 Kullanım:
# from src.data_loader import veri_yukle
# filmler, puanlar = veri_yukle()
# ============================================

# ----- Gerekli Kütüphaneleri İçe Aktar -----

# pandas: Tablo verilerini okumak ve işlemek için
# pd kısaltması yaygın kullanılır
import pandas as pd

# numpy: Sayısal işlemler için
# np kısaltması yaygın kullanılır
import numpy as np

# os: Dosya yolu işlemleri için
import os

# typing: Tip belirtmek için (kodun anlaşılırlığı için)
from typing import Tuple, Optional


# ============================================
# 📋 Sabit Değerler
# ============================================

# Varsayılan veri seti yolu
# Bu projenin ana klasöründen data klasörüne gider
VARSAYILAN_VERI_YOLU = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ml-latest-small"
)


def veri_yukle(veri_yolu: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    🎯 Bu fonksiyon ne yapar?
    MovieLens veri setindeki film ve puan bilgilerini yükler.
    
    Düşün ki Netflix'in veritabanına giriyorsun ve
    tüm filmleri ve puanları alıyorsun!
    
    📥 Parametreler:
    - veri_yolu (str, opsiyonel): Veri setinin bulunduğu klasör
      Örnek: '/home/user/data/ml-latest-small'
      Verilmezse varsayılan yol kullanılır
    
    📤 Döndürdüğü:
    - filmler (DataFrame): Film bilgileri tablosu
    - puanlar (DataFrame): Kullanıcı puanları tablosu
    
    💡 Örnek Kullanım:
    >>> filmler, puanlar = veri_yukle()
    >>> print(f"Toplam {len(filmler)} film yüklendi")
    Toplam 9742 film yüklendi
    
    🤔 Neden bu fonksiyonu yazdık?
    Her seferinde pd.read_csv yazmak yerine,
    tek bir fonksiyonla tüm veriyi yüklüyoruz.
    Hem daha temiz, hem hata kontrolü yapıyoruz!
    """
    
    # Veri yolu verilmediyse varsayılanı kullan
    # None = Hiçbir şey verilmedi demek
    if veri_yolu is None:
        veri_yolu = VARSAYILAN_VERI_YOLU
    
    # ----- Film Verilerini Yükle -----
    # movies.csv dosyasının tam yolunu oluştur
    filmler_dosya_yolu = os.path.join(veri_yolu, "movies.csv")
    
    # Dosyanın var olup olmadığını kontrol et
    if not os.path.exists(filmler_dosya_yolu):
        # Dosya yoksa hata fırlat
        raise FileNotFoundError(
            f"❌ Film dosyası bulunamadı: {filmler_dosya_yolu}\n"
            f"   Önce veri setini indirin: python data/download_data.py"
        )
    
    # CSV dosyasını oku
    # pd.read_csv: CSV dosyasını DataFrame'e çevirir
    filmler = pd.read_csv(filmler_dosya_yolu)
    
    # ----- Puan Verilerini Yükle -----
    # ratings.csv dosyasının tam yolunu oluştur
    puanlar_dosya_yolu = os.path.join(veri_yolu, "ratings.csv")
    
    # Dosyanın var olup olmadığını kontrol et
    if not os.path.exists(puanlar_dosya_yolu):
        raise FileNotFoundError(
            f"❌ Puan dosyası bulunamadı: {puanlar_dosya_yolu}\n"
            f"   Önce veri setini indirin: python data/download_data.py"
        )
    
    # CSV dosyasını oku
    puanlar = pd.read_csv(puanlar_dosya_yolu)
    
    # Kullanıcıya bilgi ver
    print(f"✅ Veriler başarıyla yüklendi!")
    print(f"   🎬 Film sayısı: {len(filmler):,}")
    print(f"   ⭐ Puan sayısı: {len(puanlar):,}")
    print(f"   👤 Kullanıcı sayısı: {puanlar['userId'].nunique():,}")
    
    # Her iki tabloyu da döndür
    return filmler, puanlar


def etiketleri_yukle(veri_yolu: Optional[str] = None) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Kullanıcıların filmlere verdiği etiketleri yükler.
    
    Etiket nedir? Kullanıcıların filmler hakkında yazdığı
    kısa notlar. Örnek: "funny", "romantic", "classic"
    
    📥 Parametreler:
    - veri_yolu (str, opsiyonel): Veri setinin bulunduğu klasör
    
    📤 Döndürdüğü:
    - etiketler (DataFrame): Etiket bilgileri tablosu
    
    💡 Örnek Kullanım:
    >>> etiketler = etiketleri_yukle()
    >>> print(etiketler.head())
    
    🤔 Neden bu fonksiyonu yazdık?
    İçerik tabanlı filtreleme için etiketler çok yararlı!
    Bir filmin hangi kategoride olduğunu anlamaya yardımcı olur.
    """
    
    # Varsayılan yolu kullan
    if veri_yolu is None:
        veri_yolu = VARSAYILAN_VERI_YOLU
    
    # Etiket dosyasının yolunu oluştur
    etiketler_dosya_yolu = os.path.join(veri_yolu, "tags.csv")
    
    # Dosya kontrolü
    if not os.path.exists(etiketler_dosya_yolu):
        raise FileNotFoundError(
            f"❌ Etiket dosyası bulunamadı: {etiketler_dosya_yolu}"
        )
    
    # CSV dosyasını oku
    etiketler = pd.read_csv(etiketler_dosya_yolu)
    
    print(f"✅ Etiketler yüklendi: {len(etiketler):,} adet")
    
    return etiketler


def linkleri_yukle(veri_yolu: Optional[str] = None) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    IMDB ve TMDB linklerini yükler.
    
    Bu linkler filmlerin resmi web sitelerindeki
    ID'lerini içerir. Poster indirmek için kullanılabilir!
    
    📥 Parametreler:
    - veri_yolu (str, opsiyonel): Veri setinin bulunduğu klasör
    
    📤 Döndürdüğü:
    - linkler (DataFrame): Link bilgileri tablosu
    """
    
    if veri_yolu is None:
        veri_yolu = VARSAYILAN_VERI_YOLU
    
    linkler_dosya_yolu = os.path.join(veri_yolu, "links.csv")
    
    if not os.path.exists(linkler_dosya_yolu):
        raise FileNotFoundError(
            f"❌ Link dosyası bulunamadı: {linkler_dosya_yolu}"
        )
    
    linkler = pd.read_csv(linkler_dosya_yolu)
    
    print(f"✅ Linkler yüklendi: {len(linkler):,} adet")
    
    return linkler


def kullanici_film_matrisi_olustur(puanlar: pd.DataFrame) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Kullanıcı-Film puan matrisini oluşturur.
    
    Bu matris şöyle görünür:
    
            Film1  Film2  Film3  Film4
    User1    5.0    3.0    NaN    4.0
    User2    NaN    4.0    5.0    NaN
    User3    3.0    NaN    4.0    5.0
    
    NaN = Kullanıcı o filmi izlememiş/puanlamamış
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu (userId, movieId, rating sütunları)
    
    📤 Döndürdüğü:
    - matris (DataFrame): Kullanıcı-Film puan matrisi
    
    💡 Örnek Kullanım:
    >>> filmler, puanlar = veri_yukle()
    >>> matris = kullanici_film_matrisi_olustur(puanlar)
    >>> print(matris.shape)
    (610, 9724)  # 610 kullanıcı x 9724 film
    
    🤔 Neden bu fonksiyonu yazdık?
    İşbirlikçi filtreleme algoritmaları bu matris
    formatındaki veriyle çalışır. Her satır bir kullanıcı,
    her sütun bir film. Hücrelerde puanlar var.
    """
    
    # pivot_table: Uzun formattaki veriyi geniş formata çevirir
    # 
    # Giriş (uzun format):
    # userId  movieId  rating
    #   1       1        5.0
    #   1       2        3.0
    #   2       1        4.0
    #
    # Çıkış (geniş format - matris):
    #         Film1  Film2
    # User1    5.0    3.0
    # User2    4.0    NaN
    
    matris = puanlar.pivot_table(
        index='userId',      # Satırlar: Kullanıcılar
        columns='movieId',   # Sütunlar: Filmler
        values='rating'      # Değerler: Puanlar
    )
    
    # Matris boyutunu göster
    kullanici_sayisi, film_sayisi = matris.shape
    print(f"✅ Kullanıcı-Film matrisi oluşturuldu!")
    print(f"   📊 Boyut: {kullanici_sayisi:,} kullanıcı × {film_sayisi:,} film")
    
    # Seyreklik (sparsity) hesapla
    # Seyreklik: Boş hücrelerin oranı
    toplam_hucre = kullanici_sayisi * film_sayisi
    dolu_hucre = puanlar.shape[0]
    seyreklik = 1 - (dolu_hucre / toplam_hucre)
    print(f"   🔲 Seyreklik: %{seyreklik * 100:.2f} (boş hücre oranı)")
    
    return matris


def veri_ozeti_goster(filmler: pd.DataFrame, puanlar: pd.DataFrame):
    """
    🎯 Bu fonksiyon ne yapar?
    Veri seti hakkında özet bilgileri ekrana yazdırır.
    
    📥 Parametreler:
    - filmler (DataFrame): Film tablosu
    - puanlar (DataFrame): Puan tablosu
    
    🤔 Neden bu fonksiyonu yazdık?
    Veriyi anlamak için ilk adım, onun özetine bakmaktır!
    """
    
    print("\n" + "=" * 60)
    print("📊 VERİ SETİ ÖZETİ")
    print("=" * 60)
    
    # Film istatistikleri
    print("\n🎬 FİLM BİLGİLERİ")
    print("-" * 40)
    print(f"   Toplam film: {len(filmler):,}")
    
    # Tür (genre) analizi
    # Her filmin türleri | ile ayrılmış
    # Örnek: "Action|Adventure|Comedy"
    tum_turler = filmler['genres'].str.split('|').explode()
    benzersiz_turler = tum_turler.nunique()
    en_populer_tur = tum_turler.value_counts().head(1)
    
    print(f"   Benzersiz tür sayısı: {benzersiz_turler}")
    print(f"   En popüler tür: {en_populer_tur.index[0]} ({en_populer_tur.values[0]:,} film)")
    
    # Puan istatistikleri
    print("\n⭐ PUAN BİLGİLERİ")
    print("-" * 40)
    print(f"   Toplam puan: {len(puanlar):,}")
    print(f"   Kullanıcı sayısı: {puanlar['userId'].nunique():,}")
    print(f"   Puanlanan film sayısı: {puanlar['movieId'].nunique():,}")
    print(f"   Ortalama puan: {puanlar['rating'].mean():.2f}")
    print(f"   Puan aralığı: {puanlar['rating'].min()} - {puanlar['rating'].max()}")
    
    # Kullanıcı başına ortalama
    puanlar_per_kullanici = len(puanlar) / puanlar['userId'].nunique()
    print(f"   Kullanıcı başına ortalama puan: {puanlar_per_kullanici:.1f}")
    
    print("\n" + "=" * 60)


# ============================================
# 🧪 Test Kodu
# ============================================
# Bu kısım dosya doğrudan çalıştırıldığında çalışır

if __name__ == "__main__":
    # Test: Veri yükleme
    print("🧪 Veri yükleme testi başlıyor...\n")
    
    try:
        # Veriyi yükle
        filmler, puanlar = veri_yukle()
        
        # Özet göster
        veri_ozeti_goster(filmler, puanlar)
        
        # Matris oluştur
        print("\n" + "-" * 40)
        matris = kullanici_film_matrisi_olustur(puanlar)
        
        print("\n✅ Tüm testler başarılı!")
        
    except FileNotFoundError as hata:
        print(f"⚠️ {hata}")
        print("\nÖnce veri setini indirin:")
        print("   python data/download_data.py")
