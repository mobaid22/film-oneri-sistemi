# ============================================
# 📦 Film Öneri Sistemi - Ana Paket
# ============================================
# 
# 🎯 Bu dosya ne yapar?
# Bu dosya src klasörünü bir Python paketi yapar.
# Böylece diğer dosyalardan import yapabiliriz.
#
# 💡 Örnek Kullanım:
# from src import data_loader
# from src.metrics import rmse_hesapla
# ============================================

# Paket versiyonu
# Semantic Versioning: MAJOR.MINOR.PATCH
# 0.1.0 = İlk geliştirme sürümü
__version__ = "0.1.0"

# Paket adı
__name__ = "film_oneri_sistemi"

# Yazar bilgisi
__author__ = "Film Öneri Sistemi Ekibi"

# Dışa aktarılacak modüller
# Bu liste, from src import * yapıldığında
# hangi modüllerin yükleneceğini belirler
__all__ = [
    "data_loader",
    "preprocessing", 
    "metrics",
    "similarity",
    "recommender",
    "utils",
    "models",
]

# Kullanıcıya bilgi mesajı
# Paket import edildiğinde görünür
print("🎬 Film Öneri Sistemi yüklendi!")
print(f"   Versiyon: {__version__}")
