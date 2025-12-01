# ============================================
# 🛠️ Yardımcı Fonksiyonlar Modülü (Utils)
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Projede sık kullanılan yardımcı fonksiyonları içerir.
# Tekrar tekrar yazılacak kodları tek yerde toplar.
#
# 💡 Kullanım:
# from src.utils import film_ara, populer_filmler
# ============================================

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import random


def film_ara(filmler: pd.DataFrame, arama: str, limit: int = 10) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    Film adına göre arama yapar.
    
    Düşün ki Google'da film arıyorsun,
    bu fonksiyon benzer şekilde çalışır!
    
    📥 Parametreler:
    - filmler (DataFrame): Film tablosu
    - arama (str): Aranacak kelime
      Örnek: "toy story"
    - limit (int): Maksimum sonuç sayısı
    
    📤 Döndürdüğü:
    - sonuclar (DataFrame): Eşleşen filmler
    
    💡 Örnek Kullanım:
    >>> sonuclar = film_ara(filmler, "matrix")
    >>> print(sonuclar)
       movieId                    title
    0     2571  Matrix, The (1999)
    1     6365  Matrix Reloaded, The (2003)
    """
    
    # Küçük harfe çevir (case-insensitive arama)
    arama = arama.lower()
    
    # Film adında arama kelimesi geçen filmleri bul
    # str.lower(): Tüm harfleri küçük yap
    # str.contains(): Kelime içerip içermediğini kontrol et
    maske = filmler['title'].str.lower().str.contains(arama, na=False)
    
    # Eşleşen filmleri al
    sonuclar = filmler[maske].head(limit)
    
    if len(sonuclar) == 0:
        print(f"❌ '{arama}' ile eşleşen film bulunamadı.")
    else:
        print(f"✅ {len(sonuclar)} film bulundu:")
        for _, film in sonuclar.iterrows():
            print(f"   [{film['movieId']}] {film['title']}")
    
    return sonuclar


def populer_filmleri_bul(puanlar: pd.DataFrame, 
                          filmler: pd.DataFrame,
                          min_puan_sayisi: int = 50,
                          limit: int = 10) -> pd.DataFrame:
    """
    🎯 Bu fonksiyon ne yapar?
    En popüler filmleri bulur.
    
    Popülerlik = Çok kişi tarafından izlenmiş + Yüksek puan
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu
    - filmler (DataFrame): Film tablosu
    - min_puan_sayisi (int): Minimum kaç kişi puanlamış olmalı
    - limit (int): Kaç film listelenecek
    
    📤 Döndürdüğü:
    - populer_filmler (DataFrame): Popüler film listesi
    
    💡 Örnek Kullanım:
    >>> populer = populer_filmleri_bul(puanlar, filmler, limit=5)
    """
    
    # Her filmin puan istatistiklerini hesapla
    film_istatistikleri = puanlar.groupby('movieId').agg({
        'rating': ['mean', 'count']  # Ortalama puan ve puan sayısı
    })
    
    # Sütun isimlerini düzelt
    film_istatistikleri.columns = ['ortalama_puan', 'puan_sayisi']
    film_istatistikleri = film_istatistikleri.reset_index()
    
    # Minimum puan sayısı filtresi
    film_istatistikleri = film_istatistikleri[
        film_istatistikleri['puan_sayisi'] >= min_puan_sayisi
    ]
    
    # Puana göre sırala (yüksekten düşüğe)
    film_istatistikleri = film_istatistikleri.sort_values(
        'ortalama_puan', 
        ascending=False
    )
    
    # Film bilgileriyle birleştir
    populer = film_istatistikleri.head(limit).merge(
        filmler[['movieId', 'title', 'genres']], 
        on='movieId'
    )
    
    print(f"🏆 En Popüler {limit} Film:")
    for i, film in populer.iterrows():
        print(f"   {film['title']}")
        print(f"      ⭐ {film['ortalama_puan']:.2f} ({int(film['puan_sayisi'])} kişi)")
    
    return populer


def kullanici_profili_olustur(kullanici_id: int,
                               puanlar: pd.DataFrame,
                               filmler: pd.DataFrame) -> dict:
    """
    🎯 Bu fonksiyon ne yapar?
    Kullanıcının film izleme profilini oluşturur.
    
    📥 Parametreler:
    - kullanici_id (int): Kullanıcı ID'si
    - puanlar (DataFrame): Puan tablosu
    - filmler (DataFrame): Film tablosu
    
    📤 Döndürdüğü:
    - profil (dict): Kullanıcı profili bilgileri
    """
    
    # Kullanıcının puanlarını al
    kullanici_puanlari = puanlar[puanlar['userId'] == kullanici_id]
    
    if len(kullanici_puanlari) == 0:
        return {"hata": "Kullanıcı bulunamadı!"}
    
    # Film bilgileriyle birleştir
    izlenen_filmler = kullanici_puanlari.merge(filmler, on='movieId')
    
    # Tür tercihlerini hesapla
    tum_turler = izlenen_filmler['genres'].str.split('|').explode()
    tur_dagilimi = tum_turler.value_counts().head(5)
    
    # Profil oluştur
    profil = {
        'kullanici_id': kullanici_id,
        'izlenen_film_sayisi': len(kullanici_puanlari),
        'ortalama_puan': kullanici_puanlari['rating'].mean(),
        'en_yuksek_puan': kullanici_puanlari['rating'].max(),
        'en_dusuk_puan': kullanici_puanlari['rating'].min(),
        'favori_turler': tur_dagilimi.to_dict(),
        'en_sevdigi_filmler': izlenen_filmler[
            izlenen_filmler['rating'] >= 4.5
        ]['title'].head(5).tolist()
    }
    
    return profil


def profili_yazdir(profil: dict):
    """
    🎯 Bu fonksiyon ne yapar?
    Kullanıcı profilini güzel formatla ekrana yazdırır.
    """
    
    if 'hata' in profil:
        print(f"❌ {profil['hata']}")
        return
    
    print("\n" + "=" * 50)
    print(f"👤 KULLANICI PROFİLİ (ID: {profil['kullanici_id']})")
    print("=" * 50)
    
    print(f"\n📊 İstatistikler:")
    print(f"   İzlenen film: {profil['izlenen_film_sayisi']}")
    print(f"   Ortalama puan: {profil['ortalama_puan']:.2f}")
    print(f"   Puan aralığı: {profil['en_dusuk_puan']} - {profil['en_yuksek_puan']}")
    
    print(f"\n🎭 Favori Türler:")
    for tur, sayi in profil['favori_turler'].items():
        print(f"   {tur}: {sayi} film")
    
    print(f"\n❤️ En Sevdiği Filmler:")
    for film in profil['en_sevdigi_filmler']:
        print(f"   • {film}")
    
    print("=" * 50)


def rastgele_kullanici_sec(puanlar: pd.DataFrame, 
                           min_puan: int = 20) -> int:
    """
    🎯 Bu fonksiyon ne yapar?
    Test için rastgele bir kullanıcı seçer.
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu
    - min_puan (int): Kullanıcının minimum puan sayısı
    
    📤 Döndürdüğü:
    - kullanici_id (int): Seçilen kullanıcı ID'si
    """
    
    # Yeterli puanı olan kullanıcıları bul
    puan_sayilari = puanlar.groupby('userId').size()
    uygun_kullanicilar = puan_sayilari[puan_sayilari >= min_puan].index.tolist()
    
    if len(uygun_kullanicilar) == 0:
        raise ValueError(f"❌ En az {min_puan} puan veren kullanıcı yok!")
    
    # Rastgele seç
    kullanici_id = random.choice(uygun_kullanicilar)
    
    print(f"🎲 Rastgele kullanıcı seçildi: {kullanici_id}")
    
    return kullanici_id


def film_turu_getir(film_id: int, filmler: pd.DataFrame) -> List[str]:
    """
    🎯 Bu fonksiyon ne yapar?
    Bir filmin türlerini liste olarak döndürür.
    
    📥 Parametreler:
    - film_id (int): Film ID'si
    - filmler (DataFrame): Film tablosu
    
    📤 Döndürdüğü:
    - turler (list): Tür listesi
      Örnek: ["Action", "Adventure", "Sci-Fi"]
    """
    
    film = filmler[filmler['movieId'] == film_id]
    
    if len(film) == 0:
        return []
    
    turler_str = film['genres'].values[0]
    
    if turler_str == "(no genres listed)":
        return []
    
    turler = turler_str.split('|')
    
    return turler


def film_yili_getir(film_adi: str) -> Optional[int]:
    """
    🎯 Bu fonksiyon ne yapar?
    Film adından yapım yılını çıkarır.
    
    MovieLens formatı: "Film Adı (2019)"
    
    📥 Parametreler:
    - film_adi (str): Film adı
      Örnek: "Toy Story (1995)"
    
    📤 Döndürdüğü:
    - yil (int veya None): Yapım yılı
    """
    
    import re
    
    # Parantez içindeki 4 haneli yılı bul
    # \((\d{4})\) : (1995) formatını yakalar
    eslesme = re.search(r'\((\d{4})\)', film_adi)
    
    if eslesme:
        return int(eslesme.group(1))
    
    return None


def puan_dagilimi_goster(puanlar: pd.DataFrame):
    """
    🎯 Bu fonksiyon ne yapar?
    Puan dağılımını konsola yazdırır.
    
    📥 Parametreler:
    - puanlar (DataFrame): Puan tablosu
    """
    
    print("\n📊 PUAN DAĞILIMI")
    print("-" * 30)
    
    dagilim = puanlar['rating'].value_counts().sort_index()
    
    toplam = len(puanlar)
    
    for puan, sayi in dagilim.items():
        yuzde = sayi / toplam * 100
        cubuk = "█" * int(yuzde / 2)  # Görsel çubuk
        print(f"   {puan}: {cubuk} {sayi:,} ({yuzde:.1f}%)")


def seyreklik_hesapla(matris: np.ndarray) -> float:
    """
    🎯 Bu fonksiyon ne yapar?
    Matrisin seyreklik (sparsity) oranını hesaplar.
    
    Seyreklik = Boş hücrelerin oranı
    
    📥 Parametreler:
    - matris: Kullanıcı-Film puan matrisi
    
    📤 Döndürdüğü:
    - seyreklik (float): 0-1 arası (1 = çok seyrek)
    """
    
    toplam_hucre = matris.size
    dolu_hucre = np.count_nonzero(~np.isnan(matris))
    
    seyreklik = 1 - (dolu_hucre / toplam_hucre)
    
    return float(seyreklik)


def tablo_boyutu_yazdir(df: pd.DataFrame, ad: str = "Tablo"):
    """
    🎯 Bu fonksiyon ne yapar?
    DataFrame boyutunu formatlanmış şekilde yazdırır.
    """
    
    satirlar, sutunlar = df.shape
    bellek = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    
    print(f"📊 {ad}: {satirlar:,} satır × {sutunlar} sütun ({bellek:.2f} MB)")


def ilerleme_goster(mevcut: int, toplam: int, aciklama: str = ""):
    """
    🎯 Bu fonksiyon ne yapar?
    Konsola ilerleme durumu yazdırır.
    
    📥 Parametreler:
    - mevcut: Mevcut adım sayısı
    - toplam: Toplam adım sayısı
    - aciklama: İşlem açıklaması
    """
    
    yuzde = (mevcut / toplam) * 100
    cubuk_uzunlugu = 20
    dolu = int(cubuk_uzunlugu * yuzde / 100)
    bos = cubuk_uzunlugu - dolu
    
    cubuk = "█" * dolu + "░" * bos
    
    print(f"\r   [{cubuk}] {yuzde:.1f}% - {aciklama}", end="", flush=True)
    
    if mevcut == toplam:
        print()  # Satır sonu


def timestamp_to_tarih(timestamp: int) -> str:
    """
    🎯 Bu fonksiyon ne yapar?
    Unix timestamp'ı okunabilir tarihe çevirir.
    
    📥 Parametreler:
    - timestamp (int): Unix timestamp
      Örnek: 964982703
    
    📤 Döndürdüğü:
    - tarih (str): Formatlanmış tarih
      Örnek: "2000-07-30 18:45:03"
    """
    
    from datetime import datetime
    
    tarih = datetime.fromtimestamp(timestamp)
    
    return tarih.strftime("%Y-%m-%d %H:%M:%S")


# ============================================
# 🧪 Test Kodu
# ============================================

if __name__ == "__main__":
    print("🧪 Utils modülü testi\n")
    
    # Tarih dönüşüm testi
    print("⏰ Timestamp testi:")
    timestamp = 964982703
    tarih = timestamp_to_tarih(timestamp)
    print(f"   {timestamp} → {tarih}")
    
    # Yıl çıkarma testi
    print("\n📅 Film yılı testi:")
    filmler = ["Toy Story (1995)", "Matrix (1999)", "Test Film"]
    for film in filmler:
        yil = film_yili_getir(film)
        print(f"   {film} → {yil}")
    
    print("\n✅ Utils testleri tamamlandı!")
