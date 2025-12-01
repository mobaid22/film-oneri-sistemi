# ============================================
# 📦 Film Öneri Sistemi - Kurulum Dosyası
# ============================================
# 
# Bu dosya projeyi bir Python paketi olarak
# kurmanızı sağlar.
#
# Kurulum: pip install -e .
# ============================================

# setuptools: Python paket kurulumu için gerekli araçlar
from setuptools import setup, find_packages

# Açıklama dosyasını oku
# README.md dosyasından proje açıklamasını alıyoruz
with open("README.md", "r", encoding="utf-8") as fh:
    # Dosyanın tüm içeriğini oku
    long_description = fh.read()

# setup() fonksiyonu paketin tüm bilgilerini tanımlar
setup(
    # Paket adı - pip install yaparken kullanılır
    name="film-oneri-sistemi",
    
    # Versiyon numarası
    # 0.1.0 = İlk geliştirme sürümü
    version="0.1.0",
    
    # Yazar bilgileri
    author="Film Öneri Sistemi Ekibi",
    author_email="ornek@email.com",
    
    # Kısa açıklama (bir satırlık)
    description="Başlangıç seviyesinde, Türkçe açıklamalı film öneri sistemi",
    
    # Uzun açıklama (README.md'den alınıyor)
    long_description=long_description,
    
    # Uzun açıklamanın formatı (Markdown)
    long_description_content_type="text/markdown",
    
    # Proje web sitesi (GitHub sayfası)
    url="https://github.com/mobaid22/film-oneri-sistemi",
    
    # Projenin kategorileri ve özellikleri
    classifiers=[
        # Geliştirme aşaması
        "Development Status :: 3 - Alpha",
        
        # Hedef kitle
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        
        # Konu
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        
        # Programlama dili ve versiyon
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Lisans
        "License :: OSI Approved :: MIT License",
        
        # İşletim sistemi
        "Operating System :: OS Independent",
        
        # Dil
        "Natural Language :: Turkish",
    ],
    
    # Kaynak kodun bulunduğu klasör
    package_dir={"": "."},
    
    # Tüm paketleri otomatik bul
    packages=find_packages(where="."),
    
    # Minimum Python versiyonu
    python_requires=">=3.8",
    
    # Gerekli kütüphaneler
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scikit-surprise>=1.1.3",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "requests>=2.28.0",
        "tqdm>=4.64.0",
    ],
    
    # Opsiyonel bağımlılıklar
    extras_require={
        # Geliştirme için gerekli ekstra kütüphaneler
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
        ],
    },
    
    # Anahtar kelimeler (arama için)
    keywords="film, öneri, recommendation, machine-learning, türkçe, eğitim",
)
