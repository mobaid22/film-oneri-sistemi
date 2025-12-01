# 📚 Temel Kavramlar

Bu belge, makine öğrenmesi ve öneri sistemlerinde kullanılan temel kavramları açıklar. Her kavram, 10 yaşında birine anlatır gibi basit bir dille açıklanmıştır.

---

## 📖 İçindekiler

1. [Overfitting (Aşırı Öğrenme)](#1-overfitting-aşırı-öğrenme)
2. [Underfitting (Yetersiz Öğrenme)](#2-underfitting-yetersiz-öğrenme)
3. [Bias-Variance Tradeoff](#3-bias-variance-tradeoff)
4. [Cross-Validation (Çapraz Doğrulama)](#4-cross-validation-çapraz-doğrulama)
5. [Train/Test Split (Eğitim/Test Bölmesi)](#5-traintest-split)
6. [Type 1 ve Type 2 Hatalar](#6-type-1-ve-type-2-hatalar)
7. [Regularization (Düzenlileştirme)](#7-regularization-düzenlileştirme)
8. [Feature Engineering (Özellik Mühendisliği)](#8-feature-engineering-özellik-mühendisliği)

---

## 1. Overfitting (Aşırı Öğrenme)

### 🤔 Nedir?

Modelin eğitim verisini **çok iyi** ezberlemesi, ama yeni verilerde kötü sonuç vermesi.

### 📚 Gerçek Hayat Örneği

Düşün ki bir sınava hazırlanıyorsun:

**Aşırı Öğrenme Senaryosu:**
- Sadece örnek soruların cevaplarını ezberledin
- Örnek: "Soru 5'in cevabı C"
- Sınavda aynı sorular çıkarsa: 💯 100 puan!
- Farklı sorular çıkarsa: 📉 0 puan!

**İyi Öğrenme Senaryosu:**
- Konuyu anladın ve mantığını kavradın
- Sınavda hangi soru çıkarsa çıksın: 📈 80 puan!

### 🎬 Film Öneri Örneği

```
Overfitting Yapan Model:
- Eğitim verisi: "Kullanıcı 5, Film 100'e 4.5 puan verdi"
- Model öğrendi: Kullanıcı 5 + Film 100 = 4.5

Test verisi: "Kullanıcı 5, Film 200?"
- Model: "Bilmiyorum, böyle bir örnek görmedim" 😕
```

### 🔧 Nasıl Anlarız?

```
Eğitim Hatası: Çok düşük (0.1)
Test Hatası:   Çok yüksek (2.5)
        ↓
    OVERFITTING VAR! ⚠️
```

### 💊 Çözümler

1. **Daha fazla veri** topla
2. **Regularization** kullan (L1, L2)
3. **Dropout** uygula
4. **Daha basit model** seç
5. **Early stopping** kullan

---

## 2. Underfitting (Yetersiz Öğrenme)

### 🤔 Nedir?

Modelin veriyi yeterince öğrenememesi. Hem eğitim hem test verisinde kötü sonuç.

### 📚 Gerçek Hayat Örneği

Düşün ki bir sınava hazırlanıyorsun:

**Yetersiz Öğrenme Senaryosu:**
- Sadece 1 saat çalıştın
- Konunun yarısını bile okumadın
- Hem örnek sorularda hem sınavda: 📉 30 puan!

### 🎬 Film Öneri Örneği

```
Underfitting Yapan Model:
- Her filme aynı puanı tahmin ediyor: 3.5
- Kullanıcının tercihlerini hiç öğrenmemiş
- "Aksiyon sever" veya "Romantik sever" ayrımı yok
```

### 🔧 Nasıl Anlarız?

```
Eğitim Hatası: Yüksek (1.5)
Test Hatası:   Yüksek (1.6)
        ↓
    UNDERFITTING VAR! ⚠️
```

### 💊 Çözümler

1. **Daha karmaşık model** kullan
2. **Daha fazla özellik** ekle
3. **Daha uzun eğit**
4. **Regularization'ı azalt**

---

## 3. Bias-Variance Tradeoff

### 🤔 Nedir?

**Bias (Önyargı)** ve **Variance (Varyans)** arasındaki denge.

### 📊 Görselleştirme

```
       ┌─────────────────────────────────────────┐
       │                                         │
  Hata │   \                               /     │
       │    \   Toplam Hata               /      │
       │     \     ___________           /       │
       │      \   /           \         /        │
       │       \_/             \_______/         │
       │        \                                │
       │         \  Bias                         │
       │          \_________                     │
       │                    \_______             │
       │                            Variance     │
       │                                         │
       └─────────────────────────────────────────┘
              Basit ◄─────────────────► Karmaşık
                     Model Karmaşıklığı
```

### 🎯 Hedef

İkisinin de düşük olduğu "tatlı noktayı" bulmak!

| Durum | Bias | Variance | Sonuç |
|-------|------|----------|-------|
| Basit model | Yüksek | Düşük | Underfitting |
| Karmaşık model | Düşük | Yüksek | Overfitting |
| **Dengeli model** | **Orta** | **Orta** | **İdeal!** |

---

## 4. Cross-Validation (Çapraz Doğrulama)

### 🤔 Nedir?

Veriyi birden fazla şekilde bölerek modelin performansını daha güvenilir ölçme yöntemi.

### 📊 K-Fold Cross-Validation

```
K = 5 örneği:

Fold 1: [TEST] [Eğitim] [Eğitim] [Eğitim] [Eğitim]
Fold 2: [Eğitim] [TEST] [Eğitim] [Eğitim] [Eğitim]
Fold 3: [Eğitim] [Eğitim] [TEST] [Eğitim] [Eğitim]
Fold 4: [Eğitim] [Eğitim] [Eğitim] [TEST] [Eğitim]
Fold 5: [Eğitim] [Eğitim] [Eğitim] [Eğitim] [TEST]

Final Skor = Ortalama(Fold1, Fold2, Fold3, Fold4, Fold5)
```

### ✅ Avantajları

1. Tüm veri hem eğitim hem test için kullanılır
2. Daha güvenilir performans tahmini
3. Modelin genelleme yeteneğini ölçer

### 📝 Python Kodu

```python
from sklearn.model_selection import cross_val_score

# 5-Fold Cross Validation
skorlar = cross_val_score(model, X, y, cv=5)
print(f"Ortalama: {skorlar.mean():.2f} (+/- {skorlar.std()*2:.2f})")
```

---

## 5. Train/Test Split

### 🤔 Nedir?

Veriyi eğitim ve test setlerine ayırma işlemi.

### 📊 Tipik Bölme Oranları

```
Veri Seti (1000 örnek)
├── Eğitim Seti: 800 örnek (80%)
│   └── Modeli EĞİTMEK için kullanılır
│
└── Test Seti: 200 örnek (20%)
    └── Modeli DEĞERLENDİRMEK için kullanılır
```

### ⚠️ Önemli Kurallar

1. **Rastgele bölme yap** - Sıralı bölme yapma!
2. **Sızıntı olmasın** - Test verisi eğitimi etkilemesin
3. **Temsili olsun** - Her sınıf orantılı olmalı

### 📝 Python Kodu

```python
from sklearn.model_selection import train_test_split

X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # %20 test
    random_state=42   # Tekrarlanabilirlik
)
```

---

## 6. Type 1 ve Type 2 Hatalar

### 🤔 Nedir?

Sınıflandırma problemlerinde yapılan iki tür hata.

### 📊 Karışıklık Matrisi

```
                    Gerçek Durum
                   Pozitif  Negatif
                 ┌─────────┬─────────┐
     Pozitif     │   TP    │   FP    │  ← Type 1 Error
  Tahmin         │ (Doğru) │ (HATA!) │    (False Positive)
                 ├─────────┼─────────┤
     Negatif     │   FN    │   TN    │  ← Type 2 Error
                 │ (HATA!) │ (Doğru) │    (False Negative)
                 └─────────┴─────────┘
```

### 🎬 Film Öneri Örneği

| Hata Türü | Açıklama | Film Örneği |
|-----------|----------|-------------|
| **Type 1 (FP)** | Sevmeyeceği film önerildi | "Korku filmi sevmiyorsun ama önerdik" |
| **Type 2 (FN)** | Seveceği film önerilmedi | "Aksiyon severdin ama Matrix'i önermedi" |

### 🏥 Tıbbi Örnek (Daha Kritik!)

| Hata Türü | Tıbbi Örnek | Sonuç |
|-----------|-------------|-------|
| **Type 1** | Sağlıklı kişiye "hastasın" demek | Gereksiz tedavi 😟 |
| **Type 2** | Hasta kişiye "sağlıklısın" demek | Tedavi edilmeme! 💀 |

---

## 7. Regularization (Düzenlileştirme)

### 🤔 Nedir?

Modelin karmaşıklığını kontrol ederek overfitting'i önleme tekniği.

### 📊 Türleri

#### L1 Regularization (Lasso)
```
Kayıp = Hata + λ × Σ|w|

- Bazı ağırlıkları TAM SIFIR yapar
- Özellik seçimi yapar
- Sparse (seyrek) modeller üretir
```

#### L2 Regularization (Ridge)
```
Kayıp = Hata + λ × Σw²

- Ağırlıkları KÜÇÜLTÜR (sıfırlamaz)
- Tüm özellikler kalır
- Daha stabil modeller
```

### 🎚️ Lambda (λ) Parametresi

```
λ = 0    → Regularization yok, overfitting riski
λ küçük  → Hafif regularization
λ büyük  → Güçlü regularization, underfitting riski
```

---

## 8. Feature Engineering (Özellik Mühendisliği)

### 🤔 Nedir?

Ham veriden modelin öğrenebileceği anlamlı özellikler oluşturma.

### 🎬 Film Öneri Örneği

**Ham Veri:**
```
Kullanıcı puanları: [4, 5, 3, 5, 4]
Film türleri: "Action|Adventure|Sci-Fi"
```

**Çıkarılan Özellikler:**
```
ortalama_puan: 4.2
puan_sayisi: 5
en_sevilen_tur: "Action"
tur_cesitliligi: 3
son_izleme_gunleri: 15
```

### 📝 Yaygın Teknikler

| Teknik | Açıklama | Örnek |
|--------|----------|-------|
| **Binning** | Sürekli → Kategorik | Yaş → "Genç/Orta/Yaşlı" |
| **Scaling** | Normalize etme | 0-5 → 0-1 arası |
| **Encoding** | Kategorik → Sayısal | "Action" → 1 |
| **Aggregation** | Gruplama | Kullanıcı başına ortalama |
| **Interaction** | Özellik çarpımı | Yaş × Cinsiyet |

---

## 📚 Özet Tablo

| Kavram | Basit Açıklama | Öneri Sisteminde |
|--------|----------------|------------------|
| Overfitting | Ezberleme | Eğitim verisini ezberler |
| Underfitting | Yetersiz öğrenme | Kalıpları yakalayamaz |
| Bias-Variance | Denge | Model karmaşıklığı |
| Cross-Validation | Güvenilir test | K parçaya bölme |
| Train/Test Split | Veri bölme | %80/%20 ayrımı |
| Type 1/2 Errors | Hata türleri | Yanlış öneri |
| Regularization | Overfitting önleme | L1, L2 ekleme |
| Feature Engineering | Özellik oluşturma | Tür, puan, tarih |

---

*Bu belge Film Öneri Sistemi projesi için hazırlanmıştır.*
