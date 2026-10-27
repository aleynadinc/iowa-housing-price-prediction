# Iowa Ev Fiyatları Tahmini

Iowa eyaletinin Ames şehrindeki ev fiyatlarını 74 özellik üzerinden tahmin eden kapsamlı bir Doğrusal Regresyon projesi. **Gerçek dünya veri ön işlemenin** tüm adımlarını içerir.

## Problem Türü
Regresyon — hedef `SalePrice` (satış fiyatı, USD).

## Veri Seti
1460 ev kaydı, 80+ özellik (sayısal ve kategorik karışık). Veri **temizlenmemiş** — eksik değerler, karma türler mevcut.

## Yaklaşım

### 1. Temizleme
- **%30'dan fazla NA** olan sütunları sil (`dropna(axis=1, thresh=...)`)
- `Id` sütununu kaldır (sadece tanımlayıcı)

### 2. Train/Test Split
`random_state=0` ile bölme

### 3. Eksik Değer Doldurma (Imputation)
- **Sayısal sütunlar** → `SimpleImputer(strategy='mean')`
- **Kategorik sütunlar** → `SimpleImputer(strategy='most_frequent')`
- `.set_output(transform='pandas')` ile DataFrame çıktısı

### 4. Ön İşleme
- **Sayısal** → `StandardScaler`
- **Kategorik** → `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`

### 5. Birleştirme
`pd.concat` ile sayısal ve kategorik işlenmiş verileri birleştir

### 6. Modelleme ve Değerlendirme
- `LinearRegression`
- **Model MAE**: ~$21,700
- **Baseline MAE** (hep ortalamayı tahmin et): ~$56,700

## Kullanılan Kütüphaneler
- `pandas`, `scikit-learn`
- `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `LinearRegression`

## Ana Çıkarım
Model baseline'dan ~2.5 kat daha iyi, ancak ev fiyatları **lineer olmayan** karmaşık bir probleme sahip. İyileştirme için: sayısal görünen kategorik özellikleri tespit, regularizasyon, özellik mühendisliği, veya non-linear modeller (Random Forest, XGBoost) denenebilir.
