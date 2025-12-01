# ⚠️ Öneri Sistemleri Problemleri ve Çözümleri

Bu belge, öneri sistemlerinde karşılaşılan yaygın problemleri ve bunların çözümlerini açıklar.

---

## 📖 İçindekiler

1. [Sparsity (Seyreklik)](#1-sparsity-seyreklik)
2. [Cold Start (Soğuk Başlangıç)](#2-cold-start-soğuk-başlangıç)
3. [Scalability (Ölçeklenebilirlik)](#3-scalability-ölçeklenebilirlik)
4. [Gray Sheep (Gri Koyun)](#4-gray-sheep-gri-koyun)
5. [Shilling Attack (Sahte Puanlama)](#5-shilling-attack-sahte-puanlama)
6. [Diversity vs Accuracy (Çeşitlilik vs Doğruluk)](#6-diversity-vs-accuracy)
7. [Filter Bubble (Filtre Balonu)](#7-filter-bubble-filtre-balonu)

---

## 1. Sparsity (Seyreklik)

### 🤔 Problem Nedir?

Kullanıcı-Film matrisindeki puanların büyük çoğunluğunun **boş (eksik)** olması.

### 📊 Görselleştirme

```
Kullanıcı-Film Matrisi:

           Film1  Film2  Film3  Film4  Film5  ...  Film10000
User1      5.0    ?      ?      ?      3.0    ...  ?
User2      ?      4.0    ?      ?      ?      ...  ?
User3      ?      ?      ?      ?      ?      ...  4.5
User4      ?      ?      3.5    ?      ?      ...  ?
...        ...    ...    ...    ...    ...    ...  ...
User10000  ?      ?      ?      ?      ?      ...  ?

? = Puanlanmamış (NaN)

Seyreklik = (Boş hücreler) / (Toplam hücreler)
         = 99,990,000 / 100,000,000
         = %99.99 seyrek! 😱
```

### 😰 Neden Problem?

1. **Benzerlik hesaplanamaz**: Ortak puan yoksa benzerlik 0
2. **Tahmin yapılamaz**: Yeterli veri yok
3. **Model öğrenemez**: Örüntüler kaybolur

### 💊 Çözümler

#### 1. Matris Ayrıştırma (Matrix Factorization)

```
SVD ile daha düşük boyutlu temsil:

Orijinal: 10000 × 10000 matris (seyrek)
     ↓
Ayrıştırma:
U: 10000 × 50 (kullanıcı faktörleri)
V: 50 × 10000 (film faktörleri)

Artık boş hücreler de tahmin edilebilir!
```

#### 2. Varsayılan Değer Doldurma

```python
# Strateji 1: Ortalama ile doldur
matris.fillna(matris.mean())

# Strateji 2: Kullanıcı ortalaması ile doldur
for user in users:
    user_mean = user.ratings.mean()
    user.ratings.fillna(user_mean)

# Strateji 3: Film ortalaması ile doldur
for item in items:
    item_mean = item.ratings.mean()
    item.ratings.fillna(item_mean)
```

#### 3. Hibrit Yöntemler

İçerik tabanlı + İşbirlikçi = Daha az seyreklik etkisi

---

## 2. Cold Start (Soğuk Başlangıç)

### 🤔 Problem Nedir?

**Yeni kullanıcı** veya **yeni film** için öneri yapamamak.

### 📊 Üç Türü Var

#### A) Yeni Kullanıcı Cold Start

```
Yeni kullanıcı: Ahmet (ID: 12345)
Puanları: Hiç yok!

İşbirlikçi Filtreleme:
- Benzer kullanıcı bulunamıyor ❌
- Tahmin yapılamıyor ❌

Ne önerelim? 🤷
```

#### B) Yeni Öğe (Film) Cold Start

```
Yeni film: "Oppenheimer" (2023)
Puanları: Henüz hiç puan yok!

İşbirlikçi Filtreleme:
- Benzer film bulunamıyor ❌
- Kime önerelim? 🤷
```

#### C) Yeni Sistem Cold Start

```
Sistem yeni kuruldu:
- Kullanıcı yok
- Puan yok
- Hiçbir şey yok!

Nereden başlayalım? 🤷
```

### 💊 Çözümler

#### 1. İçerik Tabanlı Filtreleme Kullan

```
Yeni film: "Oppenheimer"
Türler: Drama, History, Biography
Yönetmen: Christopher Nolan

Benzer filmler:
- Interstellar (Nolan)
- Inception (Nolan)
- Dunkirk (Nolan)

Bu filmleri sevenlere öner! ✅
```

#### 2. Demografik Bilgi Kullan

```
Yeni kullanıcı: Ahmet
Yaş: 25
Cinsiyet: Erkek
Konum: İstanbul

Benzer demografiye sahip kullanıcıların
en sevdiği filmler:
- The Dark Knight
- Inception
- Fight Club

Bunları öner! ✅
```

#### 3. Aktif Öğrenme (Active Learning)

```
Yeni kullanıcıya sor:

"Lütfen bu filmlerden birkaçını puanlayın:"
1. Titanic
2. The Matrix
3. Toy Story
4. The Godfather
5. Forrest Gump

(Farklı türlerden, popüler filmler seçilir)
```

#### 4. Popüler Öğeler Öner

```
Yeni kullanıcıya:
→ En popüler 10 filmi öner

Bu "güvenli" bir başlangıç noktası.
Kişiselleştirme için veri toplar.
```

#### 5. Hibrit Yaklaşım

```python
def agirlik_belirle(puan_sayisi):
    if puan_sayisi < 5:
        # Cold start - İçeriğe güven
        return {"icerik": 0.8, "isbirligi": 0.2}
    elif puan_sayisi < 20:
        # Ilık - Dengeli
        return {"icerik": 0.4, "isbirligi": 0.6}
    else:
        # Yeterli veri - İşbirlikçiye güven
        return {"icerik": 0.2, "isbirligi": 0.8}
```

---

## 3. Scalability (Ölçeklenebilirlik)

### 🤔 Problem Nedir?

Kullanıcı ve öğe sayısı arttıkça sistemin **yavaşlaması**.

### 📊 Büyüklük Analizi

```
Netflix örneği:
- 200+ milyon kullanıcı
- 15,000+ film/dizi
- 3+ milyar puan

Kullanıcı tabanlı işbirlikçi:
- Her öneri için 200M kullanıcı karşılaştırması
- O(n²) karmaşıklık 😱
```

### 💊 Çözümler

#### 1. Model Tabanlı Yöntemler (SVD)

```
Bellek tabanlı: O(mn) her sorgu
Model tabanlı: O(k) her sorgu (k << m, n)

SVD ile faktörler önceden hesaplanır,
tahmin sadece çarpım: O(k)
```

#### 2. Approximate Nearest Neighbors (ANN)

```
Tüm kullanıcıları karşılaştırma yerine
yaklaşık en yakın komşuları bul:

LSH (Locality Sensitive Hashing)
Annoy (Spotify)
FAISS (Facebook)
```

#### 3. Dağıtık Hesaplama

```
┌─────────────────────────────────────┐
│           Ana Sunucu                │
├─────────────────────────────────────┤
│                                     │
│   ┌─────┐  ┌─────┐  ┌─────┐        │
│   │Node1│  │Node2│  │Node3│ ...    │
│   └─────┘  └─────┘  └─────┘        │
│                                     │
│  Kullanıcı 1-10M  10M-20M  20M-30M │
│                                     │
└─────────────────────────────────────┘

Apache Spark, Hadoop ile paralel hesaplama
```

---

## 4. Gray Sheep (Gri Koyun)

### 🤔 Problem Nedir?

Hiçbir gruba **benzemeyen** kullanıcılar.

### 📊 Örnek

```
Tipik Kullanıcılar:
├── Grup A: Aksiyon sevenler
├── Grup B: Romantik sevenler
└── Grup C: Komedi sevenler

Gri Koyun:
- Aksiyon'dan: 3 film sever
- Romantik'ten: 2 film sever
- Korku'dan: 4 film sever
- Belgesel'den: 5 film sever

Hiçbir gruba tam uymuyor! 🐑
```

### 💊 Çözümler

1. **İçerik tabanlı filtreleme** kullan
2. **Daha ince granüler** gruplar oluştur
3. **Kümeleme** yerine **sürekli benzerlik** kullan
4. **Hibrit** yöntemler

---

## 5. Shilling Attack (Sahte Puanlama)

### 🤔 Problem Nedir?

Kötü niyetli kullanıcıların **sahte puanlar** vererek sistemi manipüle etmesi.

### 📊 Saldırı Türleri

#### A) Push Attack (Yükseltme)

```
Hedef: Film X'i öne çıkarmak

Sahte Hesap 1: Film X → 5⭐
Sahte Hesap 2: Film X → 5⭐
Sahte Hesap 3: Film X → 5⭐
...
Sahte Hesap 100: Film X → 5⭐

Sonuç: Film X herkese öneriliyor! 📈
```

#### B) Nuke Attack (Düşürme)

```
Hedef: Rakip filmi kötülemek

Sahte Hesap 1: Rakip Film → 1⭐
Sahte Hesap 2: Rakip Film → 1⭐
...

Sonuç: Rakip film önerilmiyor! 📉
```

### 💊 Çözümler

#### 1. Anomali Tespiti

```python
def sahte_kullanici_tespit(kullanici):
    # Çok fazla puan verdiyse
    if kullanici.puan_sayisi > ortalama * 10:
        return True
    
    # Sadece aynı filmlere puan verdiyse
    if kullanici.puan_cesitliligi < 5:
        return True
    
    # Çok kısa sürede çok puan verdiyse
    if kullanici.puan_hizi > threshold:
        return True
    
    return False
```

#### 2. CAPTCHA ve Doğrulama

```
Puan vermeden önce:
- CAPTCHA
- Email doğrulama
- Telefon doğrulama
```

#### 3. Güven Tabanlı Ağırlıklandırma

```
Eski kullanıcı puanı: Ağırlık = 1.0
Yeni kullanıcı puanı: Ağırlık = 0.3
Şüpheli kullanıcı:    Ağırlık = 0.0
```

---

## 6. Diversity vs Accuracy

### 🤔 Problem Nedir?

Doğruluk ve çeşitlilik arasındaki **denge** sorunu.

### 📊 İkilem

```
Yüksek Doğruluk:
- Her zaman en yüksek tahmini öner
- Hep benzer filmler çıkar
- Sıkıcı! 😴

Yüksek Çeşitlilik:
- Farklı türlerden öner
- Bazıları beğenilmeyebilir
- Riskli! 😰
```

### 💊 Çözümler

#### 1. Hybrid Re-ranking

```python
def cok_amacli_siralama(oneriler):
    skor = []
    for film in oneriler:
        dogruluk = tahmin_skoru(film)
        cesitlilik = tur_cesitlilik_skoru(film, oneriler)
        yenilik = novelty_skoru(film)
        
        final = 0.6*dogruluk + 0.3*cesitlilik + 0.1*yenilik
        skor.append(final)
    
    return sorted(oneriler, key=lambda x: skor[x])
```

#### 2. MMR (Maximal Marginal Relevance)

```
Adım 1: En yüksek skorlu filmi seç
Adım 2: Seçilene EN AZ benzer ve yüksek skorlu sonrakini seç
Adım 3: Tekrarla
```

---

## 7. Filter Bubble (Filtre Balonu)

### 🤔 Problem Nedir?

Kullanıcının sadece **dar bir içerik yelpazesi** görmesi.

### 📊 Örnek

```
Başlangıç: Aksiyon filmi izledi
     ↓
Sistem: Daha fazla aksiyon önerdi
     ↓
Kullanıcı: Daha fazla aksiyon izledi
     ↓
Sistem: SADECE aksiyon öneriyor
     ↓
Kullanıcı: Başka türleri keşfedemez!

🫧 Filtre Balonu 🫧
```

### 💊 Çözümler

1. **Serendipity** (Şanslı keşif) ekle
2. **Exploration vs Exploitation** dengesi
3. **Çeşitlilik** metriği optimize et
4. **Kullanıcıya farklı** kategoriler göster

---

## 📊 Özet Tablo

| Problem | Açıklama | Ana Çözüm |
|---------|----------|-----------|
| **Sparsity** | Çok fazla boş hücre | SVD, Hybrid |
| **Cold Start** | Yeni kullanıcı/film | İçerik tabanlı, Demografik |
| **Scalability** | Büyük veri | Model tabanlı, Dağıtık |
| **Gray Sheep** | Gruplamayan kullanıcı | Hibrit, Sürekli benzerlik |
| **Shilling** | Sahte puanlama | Anomali tespiti |
| **Diversity** | Sıkıcı öneriler | MMR, Multi-objective |
| **Filter Bubble** | Dar içerik | Serendipity, Exploration |

---

*Bu belge Film Öneri Sistemi projesi için hazırlanmıştır.*
