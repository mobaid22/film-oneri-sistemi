# 🧮 Öneri Algoritmaları

Bu belge, film öneri sistemlerinde kullanılan temel algoritmaları açıklar.

---

## 📖 İçindekiler

1. [İçerik Tabanlı Filtreleme](#1-i̇çerik-tabanlı-filtreleme)
2. [İşbirlikçi Filtreleme](#2-i̇şbirlikçi-filtreleme)
3. [SVD (Singular Value Decomposition)](#3-svd)
4. [Hibrit Yöntemler](#4-hibrit-yöntemler)
5. [Algoritma Seçim Rehberi](#5-algoritma-seçim-rehberi)

---

## 1. İçerik Tabanlı Filtreleme

### 🎯 Temel Fikir

> "Aksiyon filmi seviyorsan, başka aksiyon filmleri önerelim!"

### 📊 Çalışma Mantığı

```
┌─────────────────────────────────────────────────────┐
│                İÇERİK TABANLI                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Kullanıcı A şunları izledi:                        │
│  ├── Matrix (Aksiyon, Sci-Fi) → 5⭐                 │
│  ├── Terminator (Aksiyon, Sci-Fi) → 4.5⭐           │
│  └── Dark Knight (Aksiyon) → 5⭐                    │
│                                                     │
│  Sistem analiz etti:                                │
│  ├── Aksiyon türünü seviyor: ✅                     │
│  ├── Sci-Fi türünü seviyor: ✅                      │
│  └── Yüksek puan veriyor                            │
│                                                     │
│  Öneri:                                             │
│  └── "Inception" (Aksiyon, Sci-Fi) 🎬               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 🔧 Algoritma Adımları

1. **Özellik Çıkarımı**
   ```
   Film: "Toy Story"
   Türler: Animation|Children|Comedy
   
   Özellik Vektörü: [1, 0, 1, 1, 0, 0, ...]
                    (her tür için 0 veya 1)
   ```

2. **Kullanıcı Profili Oluşturma**
   ```
   Kullanıcının izlediği filmler + Puanlar
                    ↓
   Ağırlıklı ortalama ile profil vektörü
   ```

3. **Benzerlik Hesaplama**
   ```
   Kosinüs Benzerliği = (Profil · Film) / (||Profil|| × ||Film||)
   ```

4. **Sıralama ve Öneri**
   ```
   En yüksek benzerlik skoruna sahip filmler
   ```

### ✅ Avantajları

| Avantaj | Açıklama |
|---------|----------|
| Cold Start | Yeni filmler için çalışır |
| Şeffaflık | Neden önerdiğini açıklayabilir |
| Bağımsız | Başka kullanıcı verisi gerekmez |

### ❌ Dezavantajları

| Dezavantaj | Açıklama |
|------------|----------|
| Sürpriz yok | Sadece benzer şeyler önerir |
| Özellik gerekli | İçerik özellikleri olmalı |
| Dar görüş | Keşif yapamaz |

---

## 2. İşbirlikçi Filtreleme

### 🎯 Temel Fikir

> "Sana benzer kullanıcılar şunu sevdi, sen de seversin!"

### 📊 İki Türü Var

#### A) Kullanıcı Tabanlı (User-Based)

```
┌────────────────────────────────────────┐
│         KULLANICI TABANLI              │
├────────────────────────────────────────┤
│                                        │
│  Kullanıcı A:                          │
│  Film1: 5⭐  Film2: 4⭐  Film3: ?       │
│                                        │
│  Benzer Kullanıcı B:                   │
│  Film1: 5⭐  Film2: 4⭐  Film3: 5⭐     │
│                                        │
│  Benzer Kullanıcı C:                   │
│  Film1: 4⭐  Film2: 5⭐  Film3: 4⭐     │
│                                        │
│  Tahmin (A, Film3):                    │
│  = Ağırlıklı ortalama(B ve C)          │
│  = (0.9×5 + 0.8×4) / (0.9+0.8)         │
│  = 4.5⭐                                │
│                                        │
└────────────────────────────────────────┘
```

#### B) Öğe Tabanlı (Item-Based)

```
┌────────────────────────────────────────┐
│           ÖĞE TABANLI                  │
├────────────────────────────────────────┤
│                                        │
│  Kullanıcı A:                          │
│  Film1: 5⭐  Film2: 4⭐  Film3: ?       │
│                                        │
│  Film benzerlik matrisi:               │
│  Film3 ~ Film1: 0.85                   │
│  Film3 ~ Film2: 0.72                   │
│                                        │
│  Tahmin (A, Film3):                    │
│  = (0.85×5 + 0.72×4) / (0.85+0.72)     │
│  = 4.5⭐                                │
│                                        │
└────────────────────────────────────────┘
```

### 🔧 Matematiksel Formül

**Kullanıcı Tabanlı:**
$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} sim(u,v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |sim(u,v)|}$$

**Türkçesi:**
```
Tahmin = Kullanıcı ortalaması + Benzer kullanıcıların sapmaları
```

### ✅ Avantajları

| Avantaj | Açıklama |
|---------|----------|
| Keşif | Sürpriz öneriler yapabilir |
| İçerik bağımsız | Özellik çıkarmaya gerek yok |
| Kanıtlanmış | Netflix gibi büyük şirketlerde çalışıyor |

### ❌ Dezavantajları

| Dezavantaj | Açıklama |
|------------|----------|
| Cold Start | Yeni kullanıcı/film için çalışmaz |
| Sparsity | Seyrek veride zorlanır |
| Ölçeklenme | Büyük veri setlerinde yavaş |

---

## 3. SVD (Singular Value Decomposition)

### 🎯 Temel Fikir

> "Büyük matrisi küçük parçalara ayır, kalıpları bul!"

### 📊 Matematiksel Gösterim

```
R ≈ U × Σ × V^T

R: Kullanıcı-Film puan matrisi (m × n)
U: Kullanıcı faktör matrisi (m × k)
Σ: Tekil değerler (k × k)
V: Film faktör matrisi (n × k)
```

### 🔧 Çalışma Mantığı

```
┌────────────────────────────────────────────────────┐
│                    SVD                              │
├────────────────────────────────────────────────────┤
│                                                    │
│  Orijinal Matris (1000 kullanıcı × 5000 film):     │
│                                                    │
│           Film1  Film2  Film3  ...  Film5000       │
│  User1    5.0    ?      3.0    ...  ?              │
│  User2    ?      4.0    ?      ...  4.5            │
│  ...      ...    ...    ...    ...  ...            │
│  User1000 3.0    ?      5.0    ...  ?              │
│                                                    │
│                     ↓ Ayrıştırma                   │
│                                                    │
│  U (1000×50)  ×  Σ (50×50)  ×  V^T (50×5000)       │
│                                                    │
│  Her kullanıcı    Önem      Her film              │
│  50 gizli         ağırlığı  50 gizli              │
│  faktörle                   faktörle              │
│  temsil edilir              temsil edilir         │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 💡 Gizli Faktörler Nedir?

```
Faktör 1: "Aksiyon sevgisi"
          Kullanıcı A: 0.9 (çok sever)
          Kullanıcı B: 0.2 (az sever)
          
Faktör 2: "Romantizm sevgisi"
          Kullanıcı A: 0.1 (az sever)
          Kullanıcı B: 0.8 (çok sever)
          
Film X: [0.85, 0.1] → Aksiyon ağırlıklı
Film Y: [0.1, 0.9]  → Romantik ağırlıklı

Tahmin(A, X) = 0.9×0.85 + 0.1×0.1 = 0.78 → Yüksek!
Tahmin(A, Y) = 0.9×0.1 + 0.1×0.9 = 0.18 → Düşük!
```

### 🎯 Tahmin Formülü

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i$$

```
Tahmin = Genel ortalama + Kullanıcı bias + Film bias + (Faktör çarpımı)
```

### ✅ Avantajları

| Avantaj | Açıklama |
|---------|----------|
| Verimli | Büyük matrisleri küçültür |
| Gizli kalıplar | Görünmeyen ilişkileri bulur |
| Sparsity çözümü | Seyrekliği azaltır |

### ❌ Dezavantajları

| Dezavantaj | Açıklama |
|------------|----------|
| Yorumlama zor | Faktörler ne anlama geliyor? |
| Cold Start | Yeni kullanıcı/film problemi |
| Eğitim süresi | İteratif, uzun sürebilir |

---

## 4. Hibrit Yöntemler

### 🎯 Temel Fikir

> "Birden fazla yöntemi birleştir, en iyi sonucu al!"

### 📊 Hibrit Türleri

| Tür | Açıklama | Örnek |
|-----|----------|-------|
| **Ağırlıklı** | Tahminlerin ağırlıklı ortalaması | 0.3×İçerik + 0.7×İşbirlikçi |
| **Geçişli** | Duruma göre seç | Cold start → İçerik, değilse → İşbirlikçi |
| **Karışık** | Tüm önerileri birleştir | İçerik top 5 + İşbirlikçi top 5 |
| **Basamaklı** | Bir filtrele, diğeri sırala | İşbirlikçi → Filtrele, İçerik → Sırala |

### 🔧 Ağırlıklı Hibrit Örneği

```
┌──────────────────────────────────────────────┐
│              AĞIRLIKLI HİBRİT                │
├──────────────────────────────────────────────┤
│                                              │
│  İçerik Modeli → Tahmin: 4.2                 │
│  İşbirlikçi    → Tahmin: 3.8                 │
│                                              │
│  Ağırlıklar:                                 │
│  w_içerik = 0.3                              │
│  w_işbirligi = 0.7                           │
│                                              │
│  Hibrit Tahmin:                              │
│  = 0.3 × 4.2 + 0.7 × 3.8                     │
│  = 1.26 + 2.66                               │
│  = 3.92 ⭐                                    │
│                                              │
└──────────────────────────────────────────────┘
```

### 💡 Dinamik Ağırlıklar

```python
def agirlik_belirle(kullanici):
    puan_sayisi = kullanici.puan_sayisi
    
    if puan_sayisi < 5:
        # Cold Start - İçeriğe güven
        return (0.8, 0.2)  # içerik, işbirlikçi
    elif puan_sayisi < 20:
        # Ilık - Dengeli
        return (0.4, 0.6)
    else:
        # Aktif - İşbirlikçiye güven
        return (0.2, 0.8)
```

---

## 5. Algoritma Seçim Rehberi

### 🎯 Karar Ağacı

```
                    Başla
                      │
                      ▼
         ┌────────────────────────┐
         │  Yeni kullanıcı/film   │
         │     oranı yüksek mi?   │
         └────────────────────────┘
                 /         \
               Evet        Hayır
                │            │
                ▼            ▼
    ┌─────────────────┐  ┌─────────────────────┐
    │  İçerik Tabanlı │  │  Veri seyrek mi?    │
    │  veya Hibrit    │  │  (çok NaN var mı?)  │
    └─────────────────┘  └─────────────────────┘
                               /         \
                             Evet        Hayır
                              │            │
                              ▼            ▼
              ┌─────────────────┐  ┌─────────────────┐
              │      SVD        │  │  İşbirlikçi     │
              │   (sparsity     │  │  (bellek        │
              │    çözer)       │  │   tabanlı)      │
              └─────────────────┘  └─────────────────┘
```

### 📊 Karşılaştırma Tablosu

| Kriter | İçerik | User-Based | Item-Based | SVD | Hibrit |
|--------|--------|------------|------------|-----|--------|
| Cold Start (Kullanıcı) | ✅ | ❌ | ⚠️ | ❌ | ✅ |
| Cold Start (Öğe) | ⚠️ | ⚠️ | ❌ | ❌ | ✅ |
| Seyreklik | ✅ | ⚠️ | ⚠️ | ✅ | ✅ |
| Ölçeklenebilirlik | ✅ | ❌ | ⚠️ | ✅ | ⚠️ |
| Sürpriz Keşif | ❌ | ✅ | ✅ | ✅ | ✅ |
| Açıklanabilirlik | ✅ | ✅ | ✅ | ❌ | ⚠️ |

### 🏭 Endüstri Örnekleri

| Şirket | Algoritma | Özellik |
|--------|-----------|---------|
| **Netflix** | SVD + Hibrit | Faktörizasyon + zaman faktörü |
| **Amazon** | Item-Based | "Bunu alanlar şunları da aldı" |
| **Spotify** | Hibrit | Ses özellikleri + Kullanıcı davranışı |
| **YouTube** | Derin Öğrenme | Neural collaborative filtering |

---

## 📚 Özet

1. **İçerik Tabanlı**: Film özelliklerine göre, cold start'a iyi
2. **İşbirlikçi**: Kullanıcı benzerliğine göre, keşif için iyi
3. **SVD**: Matris ayrıştırma, büyük veri için iyi
4. **Hibrit**: Hepsini birleştir, en iyi sonuç

---

*Bu belge Film Öneri Sistemi projesi için hazırlanmıştır.*
