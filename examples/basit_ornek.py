# ============================================
# 🌟 En Basit Film Öneri Örneği
# ============================================
# 
# Bu dosya film öneri sisteminin en temel
# kullanımını gösterir.
#
# Çalıştırmak için:
# python examples/basit_ornek.py
# ============================================

# Önce sistemin kök dizinini Python'a tanıtıyoruz
# Bu sayede src modüllerini import edebiliriz
import sys
import os

# Üst klasörü (proje kökü) path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Şimdi modülleri import edebiliriz
from src.recommender import FilmOneriSistemi


def main():
    """
    🎯 Ana fonksiyon
    
    Bu fonksiyon film öneri sisteminin temel
    kullanım akışını gösterir.
    """
    
    print("=" * 60)
    print("🎬 FİLM ÖNERİ SİSTEMİ - TEMEL ÖRNEK")
    print("=" * 60)
    
    # ----- Adım 1: Sistemi Oluştur -----
    print("\n📌 Adım 1: Sistem oluşturuluyor...")
    sistem = FilmOneriSistemi()
    
    # ----- Adım 2: Veriyi Yükle -----
    print("\n📌 Adım 2: Veri yükleniyor...")
    try:
        sistem.veri_yukle()
    except Exception as e:
        print(f"\n❌ Veri yüklenemedi: {e}")
        print("\nÖnce veri setini indirin:")
        print("   cd data && python download_data.py")
        return
    
    # ----- Adım 3: Modeli Eğit -----
    print("\n📌 Adım 3: Model eğitiliyor...")
    sistem.model_egit("icerik")  # İçerik tabanlı model
    
    # ----- Adım 4: Film Ara -----
    print("\n📌 Adım 4: Film arama...")
    print("\n🔍 'matrix' kelimesiyle arama:")
    sistem.film_ara("matrix")
    
    # ----- Adım 5: Popüler Filmler -----
    print("\n📌 Adım 5: Popüler filmler...")
    sistem.populer_filmler(limit=5)
    
    # ----- Adım 6: Kullanıcı Profili -----
    print("\n📌 Adım 6: Kullanıcı profili...")
    sistem.kullanici_profili(kullanici_id=1)
    
    # ----- Adım 7: Film Öner -----
    print("\n📌 Adım 7: Film önerisi...")
    oneriler = sistem.film_oner(kullanici_id=1, n=5)
    
    # ----- Adım 8: Model Değerlendirme -----
    print("\n📌 Adım 8: Model performansı...")
    sistem.modeli_degerlendir()
    
    # ----- Sonuç -----
    print("\n" + "=" * 60)
    print("✅ TEMEL ÖRNEK TAMAMLANDI!")
    print("=" * 60)
    
    print("""
    📚 Öğrendiklerimiz:
    
    1. FilmOneriSistemi() ile sistem oluşturulur
    2. veri_yukle() ile veri yüklenir
    3. model_egit("model_tipi") ile model eğitilir
    4. film_oner(kullanici_id) ile öneri alınır
    
    🔧 Farklı modeller:
    - "icerik": İçerik tabanlı
    - "kullanici": Kullanıcı tabanlı işbirlikçi
    - "film": Film tabanlı işbirlikçi
    - "svd": SVD matris ayrıştırma
    - "hibrit": Hibrit sistem (en iyisi!)
    
    💡 İpucu: Daha iyi sonuçlar için "hibrit" modelini deneyin!
    """)


if __name__ == "__main__":
    main()
