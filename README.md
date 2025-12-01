# 🎬 Film Öneri Sistemi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Başlangıç seviyesinde, Türkçe açıklamalı makine öğrenmesi projesi**

Her satırı Türkçe açıklamalı, 10 yaşındaki birine anlatır gibi hazırlanmış kapsamlı bir film öneri sistemi projesi.

---

## 📚 İçindekiler

- [🎯 Proje Hakkında](#-proje-hakkında)
- [✨ Özellikler](#-özellikler)
- [🚀 Hızlı Başlangıç](#-hızlı-başlangıç)
- [📁 Proje Yapısı](#-proje-yapısı)
- [📓 Notebook'lar](#-notebooklar)
- [🧮 Algoritmalar](#-algoritmalar)
- [📊 Metrikler](#-metrikler)
- [🔧 Kullanım](#-kullanım)
- [🧪 Testler](#-testler)
- [📖 Dokümantasyon](#-dokümantasyon)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Proje Hakkında

Bu proje, makine öğrenmesi ve öneri sistemleri konusunda **başlangıç seviyesindeki** geliştiriciler için hazırlanmıştır.

### Ne Öğreneceksiniz?

- 📊 **Veri Analizi**: MovieLens veri seti ile keşifsel veri analizi
- 🔧 **Veri Ön İşleme**: Temizleme, dönüştürme, normalizasyon
- 🧮 **Algoritmalar**: İçerik tabanlı, işbirlikçi filtreleme, SVD, hibrit sistemler
- 📏 **Metrikler**: RMSE, MAE, Precision, Recall, F1-Score
- ⚠️ **Problemler**: Cold Start, Sparsity, Scalability ve çözümleri

### Gerçek Hayat Uygulamaları

| Platform | Ne Önerir? | Nasıl? |
|----------|-----------|--------|
| **Netflix** | Film ve dizi | İzleme geçmişi + Benzer kullanıcılar |
| **Spotify** | Şarkı | Dinleme alışkanlıkları + Müzik özellikleri |
| **Amazon** | Ürün | "Bunu alanlar şunu da aldı" |
| **YouTube** | Video | İzleme geçmişi + Trendler |

---

## ✨ Özellikler

- ✅ **Tamamen Türkçe** - Her kod satırında açıklama
- ✅ **Başlangıç Dostu** - 10 yaşına anlatır gibi
- ✅ **Kapsamlı** - 5 farklı öneri algoritması
- ✅ **Eğitici** - Jupyter notebook'larla adım adım
- ✅ **Test Edilmiş** - pytest ile birim testler
- ✅ **Dokümante** - Detaylı Markdown dokümantasyonu

---

## 🚀 Hızlı Başlangıç

### 1. Depoyu Klonla

```bash
git clone https://github.com/mobaid22/film-oneri-sistemi.git
cd film-oneri-sistemi
```

### 2. Sanal Ortam Oluştur (Önerilir)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 4. Veri Setini İndir

```bash
python data/download_data.py
```

### 5. Demo Uygulamayı Çalıştır

```bash
python app.py
```

---

## 📁 Proje Yapısı

```
film-oneri-sistemi/
├── 📄 README.md                    # Bu dosya
├── 📄 requirements.txt             # Gerekli kütüphaneler
├── 📄 setup.py                     # Kurulum dosyası
├── 📄 app.py                       # CLI demo uygulaması
│
├── 📂 data/                        # Veri dosyaları
│   ├── README.md                   # Veri seti açıklaması
│   └── download_data.py            # MovieLens indirme scripti
│
├── 📂 src/                         # Kaynak kodlar
│   ├── __init__.py
│   ├── data_loader.py              # Veri yükleme
│   ├── preprocessing.py            # Ön işleme
│   ├── metrics.py                  # Performans metrikleri
│   ├── similarity.py               # Benzerlik hesaplama
│   ├── recommender.py              # Ana öneri motoru
│   ├── utils.py                    # Yardımcı fonksiyonlar
│   └── 📂 models/                  # Öneri modelleri
│       ├── base_model.py           # Temel model sınıfı
│       ├── content_based.py        # İçerik tabanlı
│       ├── collaborative.py        # İşbirlikçi filtreleme
│       ├── svd_model.py            # SVD matris ayrıştırma
│       └── hybrid.py               # Hibrit model
│
├── 📂 notebooks/                   # Jupyter notebook'lar
│   ├── 01_giris_ve_kurulum.ipynb
│   ├── 02_veri_analizi_eda.ipynb
│   ├── 03_veri_onisleme.ipynb
│   ├── 04_model_egitimi.ipynb
│   ├── 05_model_degerlendirme.ipynb
│   └── 06_problemler_cozumler.ipynb
│
├── 📂 tests/                       # Birim testler
│   ├── test_metrics.py
│   ├── test_similarity.py
│   └── test_models.py
│
├── 📂 examples/                    # Örnek kodlar
│   ├── basit_ornek.py
│   ├── svd_ornegi.py
│   └── hibrit_ornek.py
│
└── 📂 docs/                        # Dokümantasyon
    ├── kavramlar.md                # Temel kavramlar
    ├── metrikler.md                # Metrik formülleri
    ├── algoritmalar.md             # Algoritma açıklamaları
    └── problemler.md               # Problem ve çözümler
```

---

## 📓 Notebook'lar

| # | Notebook | Açıklama |
|---|----------|----------|
| 1 | `01_giris_ve_kurulum.ipynb` | Giriş, kurulum, veri indirme |
| 2 | `02_veri_analizi_eda.ipynb` | Keşifsel veri analizi, görselleştirme |
| 3 | `03_veri_onisleme.ipynb` | Veri temizleme, normalizasyon |
| 4 | `04_model_egitimi.ipynb` | Model eğitimi (SVD, KNN, vb.) |
| 5 | `05_model_degerlendirme.ipynb` | Metrikler ve performans |
| 6 | `06_problemler_cozumler.ipynb` | Cold Start, Sparsity çözümleri |

---

## 🧮 Algoritmalar

### 1. İçerik Tabanlı Filtreleme (Content-Based)

> "Aksiyon filmi seviyorsan, başka aksiyon filmleri önerelim!"

Film türleri, yönetmen, oyuncular gibi **içerik özelliklerine** göre öneri yapar.

### 2. İşbirlikçi Filtreleme (Collaborative Filtering)

> "Sana benzer kullanıcılar şu filmleri sevdi!"

**a) Kullanıcı Tabanlı**: Benzer kullanıcıları bulur
**b) Öğe Tabanlı**: Benzer filmleri bulur

### 3. SVD (Singular Value Decomposition)

> "Büyük matrisi küçük parçalara ayır, kalıpları bul!"

Matris ayrıştırma ile **gizli faktörleri** keşfeder.

### 4. Hibrit Sistem

> "Birden fazla yöntemi birleştir, en iyi sonucu al!"

İçerik + İşbirlikçi = Daha iyi öneriler!

---

## 📊 Metrikler

| Metrik | Açıklama | İyi Değer |
|--------|----------|-----------|
| **RMSE** | Kök Ortalama Kare Hata | Düşük |
| **MAE** | Ortalama Mutlak Hata | Düşük |
| **R²** | Belirlilik Katsayısı | 1'e yakın |
| **Precision** | Öneri isabetliliği | Yüksek |
| **Recall** | Yakalama oranı | Yüksek |
| **F1-Score** | P-R dengesi | Yüksek |

---

## 🔧 Kullanım

### Basit Örnek

```python
from src.recommender import FilmOneriSistemi

# Sistemi oluştur
sistem = FilmOneriSistemi()

# Veriyi yükle
sistem.veri_yukle()

# Model eğit (hibrit önerilir)
sistem.model_egit("hibrit")

# Film öner
oneriler = sistem.film_oner(kullanici_id=1, n=10)

# Sonuçları gör
for film_id, tahmin in oneriler:
    print(f"Film {film_id}: {tahmin:.2f} ⭐")
```

### CLI Uygulaması

```bash
python app.py
```

```
🎬 FİLM ÖNERİ SİSTEMİ
=====================================

1. Veri setini indir
2. Veri setini yükle
3. Model eğit
4. Film önerisi al
...

Seçiminiz: _
```

---

## 🧪 Testler

```bash
# Tüm testleri çalıştır
pytest tests/ -v

# Sadece metrik testlerini çalıştır
pytest tests/test_metrics.py -v

# Coverage raporu
pytest tests/ --cov=src
```

---

## 📖 Dokümantasyon

| Dosya | İçerik |
|-------|--------|
| [docs/kavramlar.md](docs/kavramlar.md) | Overfitting, Underfitting, Cross-Validation |
| [docs/metrikler.md](docs/metrikler.md) | RMSE, MAE, Precision, Recall formülleri |
| [docs/algoritmalar.md](docs/algoritmalar.md) | Algoritma seçim rehberi |
| [docs/problemler.md](docs/problemler.md) | Cold Start, Sparsity çözümleri |

---

## 📦 Bağımlılıklar

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
scikit-surprise>=1.1.3
matplotlib>=3.5.0
seaborn>=0.12.0
jupyter>=1.0.0
requests>=2.28.0
tqdm>=4.64.0
pytest>=7.0.0
```

---

## 🗺️ Yol Haritası

- [x] Temel proje yapısı
- [x] Veri yükleme ve ön işleme
- [x] İçerik tabanlı filtreleme
- [x] İşbirlikçi filtreleme
- [x] SVD implementasyonu
- [x] Hibrit model
- [x] Metrik hesaplama
- [x] Test coverage
- [x] Dokümantasyon
- [ ] Web arayüzü (Flask/Streamlit)
- [ ] Derin öğrenme modelleri

---

## 🤝 Katkıda Bulunma

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit'leyin (`git commit -m 'Yeni özellik eklendi'`)
4. Push'layın (`git push origin feature/YeniOzellik`)
5. Pull Request açın

---

## 📜 Lisans

Bu proje MIT lisansı altında sunulmaktadır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

## 🙏 Teşekkürler

- [MovieLens](https://movielens.org/) - Veri seti için
- [GroupLens Research](https://grouplens.org/) - Araştırma için
- Tüm katkıda bulunanlara!

---

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

*Bu proje eğitim amaçlıdır. Türkçe kaynakların artması dileğiyle!* 🇹🇷
