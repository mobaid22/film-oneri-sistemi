# ============================================
# 📥 MovieLens Veri Seti İndirme Scripti
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# MovieLens film veri setini internetten indirir
# ve kullanıma hazır hale getirir.
#
# 💡 Kullanım:
# python data/download_data.py
# ============================================

# ----- Gerekli Kütüphaneleri İçe Aktar -----

# os: Dosya ve klasör işlemleri için
# Örnek: Klasör oluşturma, dosya silme
import os

# zipfile: ZIP dosyalarını açmak için
# Veri seti ZIP olarak geliyor, açmamız lazım
import zipfile

# requests: İnternetten dosya indirmek için
# HTTP istekleri gönderir
import requests

# tqdm: İlerleme çubuğu göstermek için
# İndirme sırasında yüzde gösterir
from tqdm import tqdm

# shutil: Dosya/klasör kopyalama ve taşıma
import shutil


# ============================================
# 📋 Sabit Değerler (Değişmeyecek bilgiler)
# ============================================

# MovieLens veri setinin indirme adresi
# Bu URL GroupLens araştırma grubuna ait
DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# İndirilen dosyanın kaydedileceği yer
# __file__: Bu dosyanın bulunduğu konum
# dirname: Klasör yolunu al
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ZIP dosyasının adı
ZIP_FILENAME = "ml-latest-small.zip"


def indir_veri_seti():
    """
    🎯 Bu fonksiyon ne yapar?
    MovieLens veri setini internetten indirir.
    
    Düşün ki internetten bir film indiriyorsun,
    ama biz film yerine veri indiriyoruz!
    
    📤 Döndürdüğü:
    - str: İndirilen dosyanın tam yolu
    
    💡 Örnek Kullanım:
    >>> dosya_yolu = indir_veri_seti()
    >>> print(dosya_yolu)
    '/home/user/data/ml-latest-small.zip'
    
    🤔 Neden bu fonksiyonu yazdık?
    Veri setini her seferinde elle indirmek yerine
    otomatik indirmek daha pratik!
    """
    
    # ZIP dosyasının tam yolunu oluştur
    # Örnek: /home/user/data/ml-latest-small.zip
    zip_yolu = os.path.join(DATA_DIR, ZIP_FILENAME)
    
    # Eğer dosya zaten varsa, tekrar indirme
    # Bu gereksiz internet kullanımını önler
    if os.path.exists(zip_yolu):
        # Kullanıcıya bilgi ver
        print(f"✅ Veri seti zaten mevcut: {zip_yolu}")
        return zip_yolu
    
    # Kullanıcıya indirme başladığını bildir
    print("📥 MovieLens veri seti indiriliyor...")
    print(f"   URL: {DOWNLOAD_URL}")
    
    # HTTP GET isteği gönder
    # stream=True: Dosyayı parça parça indir (bellek tasarrufu)
    response = requests.get(DOWNLOAD_URL, stream=True)
    
    # İstek başarılı mı kontrol et
    # 200 = Başarılı, 404 = Bulunamadı, 500 = Sunucu hatası
    response.raise_for_status()  # Hata varsa exception fırlat
    
    # Dosyanın toplam boyutunu al (byte cinsinden)
    # Content-Length başlığından okuyoruz
    toplam_boyut = int(response.headers.get('content-length', 0))
    
    # İlerleme çubuğu oluştur
    # unit='iB' = Byte gösterimi
    # unit_scale=True = KB, MB olarak göster
    ilerleme_cubugu = tqdm(
        total=toplam_boyut,
        unit='iB',
        unit_scale=True,
        desc="İndiriliyor"
    )
    
    # Dosyayı diske kaydet
    # 'wb' = Write Binary (ikili yazma modu)
    with open(zip_yolu, 'wb') as dosya:
        # Veriyi 8192 byte'lık (8 KB) parçalar halinde indir
        # Bu büyük dosyaları indirirken bellek tasarrufu sağlar
        for parca in response.iter_content(chunk_size=8192):
            # Parçayı dosyaya yaz
            dosya.write(parca)
            # İlerleme çubuğunu güncelle
            ilerleme_cubugu.update(len(parca))
    
    # İlerleme çubuğunu kapat
    ilerleme_cubugu.close()
    
    # Başarı mesajı
    print(f"✅ İndirme tamamlandı: {zip_yolu}")
    
    # Dosya yolunu döndür
    return zip_yolu


def zip_ac(zip_yolu):
    """
    🎯 Bu fonksiyon ne yapar?
    ZIP dosyasını açar ve içindeki dosyaları çıkarır.
    
    Düşün ki bir hediye paketi açıyorsun,
    ama paketin içinde film verileri var!
    
    📥 Parametreler:
    - zip_yolu (str): Açılacak ZIP dosyasının yolu
      Örnek: '/home/user/data/ml-latest-small.zip'
    
    📤 Döndürdüğü:
    - str: Açılan klasörün yolu
    
    🤔 Neden bu fonksiyonu yazdık?
    Veri seti ZIP formatında geliyor,
    kullanmak için açmamız lazım!
    """
    
    # Açılacak klasörün yolunu oluştur
    # ZIP dosyasının adından .zip uzantısını çıkar
    hedef_klasor = os.path.join(DATA_DIR, "ml-latest-small")
    
    # Klasör zaten varsa, tekrar açma
    if os.path.exists(hedef_klasor):
        print(f"✅ Veri seti zaten açılmış: {hedef_klasor}")
        return hedef_klasor
    
    # Kullanıcıya bilgi ver
    print("📂 ZIP dosyası açılıyor...")
    
    # ZIP dosyasını aç
    # 'r' = Read (okuma modu)
    with zipfile.ZipFile(zip_yolu, 'r') as zip_dosyasi:
        # Tüm dosyaları çıkar
        # DATA_DIR = Çıkarılacak klasör
        zip_dosyasi.extractall(DATA_DIR)
    
    # Başarı mesajı
    print(f"✅ Açma tamamlandı: {hedef_klasor}")
    
    return hedef_klasor


def veri_seti_bilgisi_goster(klasor_yolu):
    """
    🎯 Bu fonksiyon ne yapar?
    İndirilen veri seti hakkında bilgi gösterir.
    
    Kaç film var? Kaç kullanıcı var? Kaç puan var?
    Bunları ekrana yazdırır.
    
    📥 Parametreler:
    - klasor_yolu (str): Veri setinin klasör yolu
    
    🤔 Neden bu fonksiyonu yazdık?
    Veri setini tanımak önemli!
    Ne kadar veri var bilmeliyiz.
    """
    
    # pandas kütüphanesini içe aktar
    # Veri okumak için kullanacağız
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ pandas kütüphanesi yüklü değil.")
        print("   Kurulum için: pip install pandas")
        return
    
    print("\n" + "=" * 50)
    print("📊 VERİ SETİ BİLGİLERİ")
    print("=" * 50)
    
    # movies.csv dosyasını oku
    filmler_yolu = os.path.join(klasor_yolu, "movies.csv")
    if os.path.exists(filmler_yolu):
        filmler = pd.read_csv(filmler_yolu)
        print(f"\n🎬 Film Sayısı: {len(filmler):,}")
        print(f"   Örnek filmler:")
        # İlk 3 filmi göster
        for i, satir in filmler.head(3).iterrows():
            print(f"   - {satir['title']}")
    
    # ratings.csv dosyasını oku
    puanlar_yolu = os.path.join(klasor_yolu, "ratings.csv")
    if os.path.exists(puanlar_yolu):
        puanlar = pd.read_csv(puanlar_yolu)
        print(f"\n⭐ Toplam Puan Sayısı: {len(puanlar):,}")
        print(f"   Kullanıcı Sayısı: {puanlar['userId'].nunique():,}")
        print(f"   Ortalama Puan: {puanlar['rating'].mean():.2f}")
        print(f"   Puan Aralığı: {puanlar['rating'].min()} - {puanlar['rating'].max()}")
    
    # tags.csv dosyasını oku
    etiketler_yolu = os.path.join(klasor_yolu, "tags.csv")
    if os.path.exists(etiketler_yolu):
        etiketler = pd.read_csv(etiketler_yolu)
        print(f"\n🏷️ Etiket Sayısı: {len(etiketler):,}")
    
    print("\n" + "=" * 50)


def ana_fonksiyon():
    """
    🎯 Bu fonksiyon ne yapar?
    Tüm veri indirme ve açma işlemlerini yönetir.
    
    Program çalıştırıldığında ilk bu fonksiyon çalışır.
    
    🤔 Neden bu fonksiyonu yazdık?
    Tüm adımları tek bir yerden kontrol etmek için!
    """
    
    # Hoş geldin mesajı
    print("🎬 MovieLens Veri Seti İndirme Aracı")
    print("=" * 40)
    
    try:
        # Adım 1: Veri setini indir
        zip_yolu = indir_veri_seti()
        
        # Adım 2: ZIP dosyasını aç
        klasor_yolu = zip_ac(zip_yolu)
        
        # Adım 3: Veri seti bilgilerini göster
        veri_seti_bilgisi_goster(klasor_yolu)
        
        print("\n✅ Tüm işlemler başarıyla tamamlandı!")
        print(f"📂 Veri seti konumu: {klasor_yolu}")
        print("\nŞimdi notebooks klasöründeki defterleri kullanabilirsiniz!")
        
    except requests.exceptions.RequestException as hata:
        # İnternet bağlantı hatası
        print(f"\n❌ İndirme hatası: {hata}")
        print("   İnternet bağlantınızı kontrol edin.")
        
    except zipfile.BadZipFile:
        # ZIP dosyası bozuk
        print("\n❌ ZIP dosyası bozuk!")
        print("   Dosyayı silip tekrar indirmeyi deneyin.")


# ============================================
# 🚀 Program Başlangıç Noktası
# ============================================
# 
# Bu kısım dosya doğrudan çalıştırıldığında
# (python download_data.py) çalışır.
# 
# Başka bir dosyadan import edildiğinde çalışmaz.
# ============================================

if __name__ == "__main__":
    # Ana fonksiyonu çağır
    ana_fonksiyon()
