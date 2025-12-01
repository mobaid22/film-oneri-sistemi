# 📊 Metrik Formülleri

Bu belge, öneri sistemlerinde kullanılan performans metriklerini detaylı olarak açıklar.

---

## 📖 İçindekiler

1. [RMSE (Root Mean Square Error)](#1-rmse)
2. [MAE (Mean Absolute Error)](#2-mae)
3. [MSE (Mean Square Error)](#3-mse)
4. [R² (R-Squared)](#4-r²)
5. [MAPE (Mean Absolute Percentage Error)](#5-mape)
6. [Precision (Kesinlik)](#6-precision)
7. [Recall (Duyarlılık)](#7-recall)
8. [F1-Score](#8-f1-score)
9. [NDCG (Normalized DCG)](#9-ndcg)

---

## 1. RMSE

### Root Mean Square Error (Kök Ortalama Kare Hata)

#### 📐 Formül

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Basit gösterim:
```
RMSE = √(1/n × Σ(gerçek - tahmin)²)
```

#### 🔢 Adım Adım Hesaplama

```
Gerçek puanlar:  [5, 4, 3, 5, 4]
Tahmin puanlar:  [4.5, 4.2, 2.8, 4.5, 3.5]

Adım 1: Hataları hesapla
        [0.5, -0.2, 0.2, 0.5, 0.5]

Adım 2: Karesini al
        [0.25, 0.04, 0.04, 0.25, 0.25]

Adım 3: Ortalamasını al
        (0.25 + 0.04 + 0.04 + 0.25 + 0.25) / 5 = 0.166

Adım 4: Karekök al
        √0.166 = 0.407

RMSE = 0.407 ⭐
```

#### 💡 Yorumlama

| RMSE Değeri | Yorum | Film Puanlama (1-5) İçin |
|-------------|-------|--------------------------|
| < 0.5 | Mükemmel | Ortalama yarım puandan az hata |
| 0.5 - 1.0 | İyi | Kabul edilebilir |
| 1.0 - 1.5 | Orta | İyileştirme gerekebilir |
| > 1.5 | Kötü | Model yetersiz |

#### ❓ Neden RMSE?

1. **Büyük hataları cezalandırır** - Kare almak büyük hataları daha önemli yapar
2. **Aynı birimde** - Sonuç orijinal veri birimiyle aynı (puan)
3. **Standart** - Akademide ve yarışmalarda çok kullanılır (Netflix Prize gibi)

---

## 2. MAE

### Mean Absolute Error (Ortalama Mutlak Hata)

#### 📐 Formül

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

```
MAE = 1/n × Σ|gerçek - tahmin|
```

#### 🔢 Örnek Hesaplama

```
Gerçek: [5, 4, 3]
Tahmin: [4.5, 4.0, 2.5]

Hatalar:    |5-4.5| = 0.5
            |4-4.0| = 0.0
            |3-2.5| = 0.5

MAE = (0.5 + 0.0 + 0.5) / 3 = 0.33 ⭐
```

#### 🆚 RMSE vs MAE

| Özellik | RMSE | MAE |
|---------|------|-----|
| Büyük hatalara tepki | Çok cezalandırır | Eşit davranır |
| Yorumlama | Zor | Kolay ("ortalama X hata") |
| Uç değerlere duyarlılık | Yüksek | Düşük |

---

## 3. MSE

### Mean Square Error (Ortalama Kare Hata)

#### 📐 Formül

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```
MSE = 1/n × Σ(gerçek - tahmin)²
```

#### 📝 Not

```
RMSE = √MSE
MSE = RMSE²
```

MSE, matematiksel işlemlerde (türev almak gibi) daha kullanışlıdır.

---

## 4. R²

### R-Squared (Belirlilik Katsayısı)

#### 📐 Formül

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Burada:
- $SS_{res} = \sum(y_i - \hat{y}_i)^2$ (Residual/Kalıntı Kareler Toplamı)
- $SS_{tot} = \sum(y_i - \bar{y})^2$ (Toplam Kareler Toplamı)

```
R² = 1 - (Tahmin hataları karesi toplamı) / (Ortalamadan sapmalar karesi toplamı)
```

#### 💡 Yorumlama

| R² Değeri | Yorum |
|-----------|-------|
| 1.0 | Mükemmel (tüm varyansı açıklıyor) |
| 0.8 - 1.0 | Çok iyi |
| 0.6 - 0.8 | İyi |
| 0.4 - 0.6 | Orta |
| 0.0 - 0.4 | Zayıf |
| < 0 | Ortalamadan bile kötü! |

#### 📊 Görsel Açıklama

```
R² = 0.85 demek:

"Modelimiz veriyi %85 oranında açıklıyor"

┌────────────────────────────────────────┐
│████████████████████████████████████░░░░│
│         Açıklanan: %85        │Açıklanamayan│
│         (Model sayesinde)     │   %15       │
└────────────────────────────────────────┘
```

---

## 5. MAPE

### Mean Absolute Percentage Error (Ortalama Mutlak Yüzde Hata)

#### 📐 Formül

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

```
MAPE = (100/n) × Σ|(gerçek - tahmin) / gerçek|
```

#### 🔢 Örnek

```
Gerçek: 100, Tahmin: 90
Yüzde hata = |100 - 90| / 100 = %10

MAPE = %10 demek: "Ortalama %10 hata yapıyoruz"
```

#### ⚠️ Dikkat

- Gerçek değer 0 olduğunda hesaplanamaz (sıfıra bölme!)
- Film puanları (0.5-5) için sorun olmaz

---

## 6. Precision

### Kesinlik

#### 📐 Formül

$$Precision = \frac{TP}{TP + FP}$$

```
Precision = Doğru Pozitifler / Tüm Pozitif Tahminler
          = Doğru Öneriler / Tüm Öneriler
```

#### 🎬 Film Örneği

```
Önerdiğimiz 10 film:
✅ 6 tanesi gerçekten beğenildi
❌ 4 tanesi beğenilmedi

Precision = 6 / 10 = 0.60 = %60

"Önerilerimizin %60'ı isabetli"
```

---

## 7. Recall

### Duyarlılık

#### 📐 Formül

$$Recall = \frac{TP}{TP + FN}$$

```
Recall = Doğru Pozitifler / Tüm Gerçek Pozitifler
       = Önerilen Beğenilenler / Tüm Beğenilebilecekler
```

#### 🎬 Film Örneği

```
Kullanıcının beğenebileceği 20 film var.
Bunlardan 8 tanesini önerdik ve beğendi.

Recall = 8 / 20 = 0.40 = %40

"Beğenilecek filmlerin %40'ını yakaladık"
```

---

## 8. F1-Score

### Precision ve Recall Dengesi

#### 📐 Formül

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

```
F1 = 2 × (P × R) / (P + R)
```

#### 💡 Neden Harmonik Ortalama?

```
Normal ortalama yanıltıcı olabilir:
P = 0.9, R = 0.1
Normal ortalama = (0.9 + 0.1) / 2 = 0.5 (yüksek görünüyor!)
Harmonik ortalama = 2×0.9×0.1 / (0.9+0.1) = 0.18 (gerçekçi!)
```

---

## 9. NDCG

### Normalized Discounted Cumulative Gain

Sıralama kalitesini ölçer - "İyi filmler listenin başında mı?"

#### 📐 Formül

$$DCG = \sum_{i=1}^{k} \frac{rel_i}{log_2(i+1)}$$

$$NDCG = \frac{DCG}{IDCG}$$

#### 💡 Mantık

```
Sıralama: [Film A (5⭐), Film B (3⭐), Film C (4⭐)]

Pozisyon 1: 5 / log₂(2) = 5.00
Pozisyon 2: 3 / log₂(3) = 1.89
Pozisyon 3: 4 / log₂(4) = 2.00

DCG = 5.00 + 1.89 + 2.00 = 8.89

İdeal sıralama: [5⭐, 4⭐, 3⭐]
IDCG = 5/1 + 4/1.58 + 3/2 = 5 + 2.53 + 1.5 = 9.03

NDCG = 8.89 / 9.03 = 0.98 ⭐
```

---

## 📊 Metrik Karşılaştırma Tablosu

| Metrik | Aralık | İyi Değer | Ne Ölçer? |
|--------|--------|-----------|-----------|
| RMSE | 0 - ∞ | Düşük | Tahmin hatası |
| MAE | 0 - ∞ | Düşük | Ortalama hata |
| R² | -∞ - 1 | 1'e yakın | Açıklanan varyans |
| MAPE | 0% - ∞% | Düşük | Yüzde hata |
| Precision | 0 - 1 | Yüksek | Öneri isabetliliği |
| Recall | 0 - 1 | Yüksek | Yakalama oranı |
| F1 | 0 - 1 | Yüksek | P-R dengesi |
| NDCG | 0 - 1 | 1'e yakın | Sıralama kalitesi |

---

## 🎯 Hangi Metriği Kullanalım?

| Senaryo | Önerilen Metrik |
|---------|-----------------|
| Puan tahmini | RMSE, MAE |
| Model karşılaştırma | RMSE, R² |
| Öneri listesi | Precision, Recall, F1 |
| Sıralama kalitesi | NDCG |
| İş metrikleri | MAPE (yüzde olarak) |

---

*Bu belge Film Öneri Sistemi projesi için hazırlanmıştır.*
