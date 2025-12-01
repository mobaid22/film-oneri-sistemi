# ============================================
# 📂 Data Klasörü - Veri Seti Bilgileri
# ============================================

## 🎬 MovieLens Veri Seti

Bu projede **MovieLens** veri setini kullanıyoruz. MovieLens, film öneri sistemleri için en popüler ve ücretsiz veri setidir.

### 📊 Veri Seti Detayları

| Özellik | Değer |
|---------|-------|
| **İsim** | MovieLens Latest Small |
| **Film Sayısı** | ~9,000+ |
| **Kullanıcı Sayısı** | ~600+ |
| **Puan Sayısı** | ~100,000 |
| **Puan Aralığı** | 0.5 - 5.0 (0.5'lik artışlarla) |
| **Kaynak** | [GroupLens Research](https://grouplens.org/datasets/movielens/) |

### 📁 Dosya Yapısı

Veri seti indirildikten sonra aşağıdaki dosyalar oluşur:

```
data/
├── ml-latest-small/
│   ├── movies.csv      # Film bilgileri (id, isim, türler)
│   ├── ratings.csv     # Kullanıcı puanları (kullanıcı_id, film_id, puan, zaman)
│   ├── tags.csv        # Kullanıcı etiketleri (film hakkında notlar)
│   └── links.csv       # IMDB ve TMDB linkleri
```

### 📥 Veri İndirme

Veriyi indirmek için:

```python
python data/download_data.py
```

### 🔍 Örnek Veriler

#### movies.csv
| movieId | title | genres |
|---------|-------|--------|
| 1 | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2 | Jumanji (1995) | Adventure\|Children\|Fantasy |

#### ratings.csv
| userId | movieId | rating | timestamp |
|--------|---------|--------|-----------|
| 1 | 1 | 4.0 | 964982703 |
| 1 | 3 | 4.0 | 964981247 |

### ⚠️ Önemli Notlar

1. **Boyut**: Veri seti yaklaşık 1 MB boyutundadır
2. **Güncellik**: Veri seti düzenli olarak güncellenir
3. **Lisans**: Eğitim amaçlı kullanım için ücretsizdir
4. **Encoding**: Dosyalar UTF-8 formatındadır

### 🤔 Neden MovieLens?

1. **Ücretsiz**: Akademik ve eğitim amaçlı ücretsiz
2. **Temiz**: Veriler düzgün formatlanmış
3. **Popüler**: Binlerce araştırmada kullanılmış
4. **Belgelenmiş**: İyi dokümante edilmiş
5. **Güncel**: Düzenli olarak güncelleniyor

### 📚 Kaynaklar

- [MovieLens Resmi Sayfası](https://movielens.org/)
- [GroupLens Research](https://grouplens.org/)
- [Veri Seti İndirme Sayfası](https://grouplens.org/datasets/movielens/)

---
*Bu veri seti GroupLens Research tarafından sağlanmaktadır.*
