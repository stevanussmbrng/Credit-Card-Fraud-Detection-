# Laporan Proyek _Machine Learning_ - Stevanus Sembiring

## DETEKSI PENIPUAN KARTU KREDIT MENGGUNAKAN _SUPERVISED MACHINE LEARNING_

Masifnya penggunaan teknologi saat ini, didorong adanya transformasi digital pasca terjadinya pandemi COVID-19, tidak terlepas mendorong perubahan pada pola pembayaran yang awalnya dilakukan secara tunai menjadi non-tunai (_cash-less_). Hingga kini, penggunaan mode pembayaran menggunakan kartu kredit bertambah seiring waktu di mana penggunaan kartu kredit menjadi pilihan karena tidak harus melakukan pembayaran secara tunai. Berdasarkan statistik Sistem Pembayaran Dan Infrastruktur Pasar Keuangan (SPIP) [(Bank Indonesia, 2024)](https://www.bi.go.id/id/statistik/ekonomi-keuangan/spip/Pages/SPIP-Januari-2024.aspx) jumlah pemegang kartu kredit di Indonesia sebanyak 17.727.481 unit sedangkan volume transaksinya sebesar 36.609.893 transaksi, kemudian nilai transaksi yang bersumber dari kartu kredit sebanyak Rp37,918 Miliar.

Dibuktikan dengan banyaknya pengguna dari informasi statistik Bank Indonesia, menjadikan salah satu faktor yang menjadikan kartu kredit sebagai target utama tindak pidana seperti penipuan (_fraud_) yakni transaksi tidak sah yang dilakukan oleh orang yang tidak dikenali dengan memanfaatkan adanya kebocoran data pribadi pengguna. Keamanan dari data pribadi memang menjadi tanggung jawab bersama antara pihak penyedia layanan kartu kredit (bank) dengan nasabah yang memegang kartu kredit. Sehingga pihak yang memiliki kepentingan wajib dengan sungguh-sungguh melakukan penjagaan agar data pribadi dari kartu kredit tidak dapat disalahgunakan oleh orang yang tidak bertanggung jawab. Hingga saat ini telah dilakukan berbagai cara lain untuk dapat melakukan tindakan preventif dengan melakukan otomasi agar dapat dilakukan proses pendeteksian secara sedini mungkin.

Dari banyaknya jumlah, volume dan nilai transaksi yang terjadi pada proses kartu kredit secara terus-menerus dari waktu ke waktu, maka solusi untuk dapat menangani masalah ini yakni mengembangkan sistem pendeteksian penipuan (_fraud_) yang dapat secara cepat, otomatis akurat, serta bekerja terus-menerus yakni dengan membangun model supervised machine learning. Berdasarkan penelitian yang dilakukan oleh [(Ningsih dkk., 2022)](https://doi.org/10.37012/jtik.v8i2.1306) dilakukan pembuatan model machine learning menggunakan _Decision Tree_, _Random Forest_, _Logistic Regression_, dan SVM (_Support Vector Machines_) menunjukkan bahwa model Random Forest menghasilkan nilai performa keseluruhan paling baik dibandingkan algoritma lainnya, yaitu menghasilkan akurasi sebesar 92%. Penelitian lain juga menggunakan algoritma yang sama yaitu _Random Forest_ untuk dapat mendeteksi adanya _fraud_ pada kartu kredit yang dilakukan [(Kurniawan & Yulianingsih, 2021)](https://doi.org/10.33322/kilat.v10i2.1482) di mana model yang dihasilkan dapat menghasilkan akurasi sebesar 85%.

## Business Understanding
### Problem Statements
- Bagaimana membedakan transaksi kartu kredit yang sah dengan yang penipuan?
- Bagaimana mengatasi data yang tidak seimbang pada data transaksi penipuan?
- Algoritma machine learning apa yang paling efektif untuk mendeteksi penipuan kartu kredit?

### Goals

- Mengidentifikasi transaksi kartu kredit yang merupakan penipuan dengan akurasi tinggi.
- Menentukan metode untuk mengatasi _imabalanced data_
- Meningkatkan performa model deteksi penipuan dengan menggunakan teknik tuning hyperparameter.

### Solution Statements

- Menggunakan beberapa algoritma machine learning seperti Logistic Regression, Random Forest, dan Gradient Boosting untuk mendeteksi penipuan.
- Menggunakan metode oversampling untuk mengatasi masalah _imbalanced data_
- Menggunakan metrik evaluasi yang sesuai seperti precision, recall, dan F1-score untuk memastikan model memberikan hasil yang akurat dan reliabel.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset deteksi penipuan kartu kredit yang tersedia di [Kaggle Datasets](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud). Dataset ini memiliki 1.000.000 baris dan 8 kolom dan tidak terdapat data yang kosong (_missing value_). Sehingga secara keseluruhan dataset ini sudah bershi dan tidak perlu melakukan penanganan _missing value_.  

### Variabel-variabel pada dataset adalah sebagai berikut:
- `distance_from_home`: Jarak dari rumah tempat transaksi terjadi. (Rasio)
- `distance_from_last_transaction`: Jarak dari transaksi terakhir yang terjadi. (Rasio)
- `ratio_to_median_purchase_price`: Rasio transaksi harga beli terhadap harga beli rata-rata. (Rasio)
- `repeat_retailer`: Apakah transaksi terjadi dari pengecer yang sama. (Nominal)
- `used_chip`: Apakah transaksi melalui chip (kartu kredit). (Nominal)
- `used_pin_number`: Apakah transaksi dilakukan dengan menggunakan nomor PIN. (Nominal)
- `online_order`: Transaksi tersebut merupakan pesanan online. (Nominal)
- `fraud`: Apakah transaksi tersebut fraud. (Nominal)

### Statistika Deskriptif
|                         | distance_from_home | distance_from_last_transaction | ratio_to_median_purchase_price | repeat_retailer | used_chip | used_pin_number | online_order | fraud    |
|-------------------------|:------------------:|:------------------------------:|:------------------------------:|:---------------:|:---------:|:---------------:|:------------:|---------:|
| count                   | 1000000.000000     | 1000000.000000                 | 1000000.000000                 | 1000000.000000  | 1000000.000000 | 1000000.000000 | 1000000.000000 | 1000000.000000 |
| mean                    | 26.628792          | 5.036519                       | 1.824182                       | 0.881536        | 0.350399  | 0.100608        | 0.650552     | 0.087403 |
| std                     | 65.390784          | 25.843093                      | 2.799589                       | 0.323157        | 0.477095  | 0.300809        | 0.476796     | 0.282425 |
| min                     | 0.004874           | 0.000118                       | 0.004399                       | 0.000000        | 0.000000  | 0.000000        | 0.000000     | 0.000000 |
| 25%                     | 3.878008           | 0.296671                       | 0.475673                       | 1.000000        | 0.000000  | 0.000000        | 0.000000     | 0.000000 |
| 50%                     | 9.967760           | 0.998650                       | 0.997717                       | 1.000000        | 0.000000  | 0.000000        | 1.000000     | 0.000000 |
| 75%                     | 25.743985          | 3.355748                       | 2.096370                       | 1.000000        | 1.000000  | 0.000000        | 1.000000     | 0.000000 |
| max                     | 10632.723672       | 11851.104565                   | 267.802942                     | 1.000000        | 1.000000  | 1.000000        | 1.000000     | 1.000000 |

Interpretasi
- Variabilitas Jarak: Jarak transaksi dari rumah dan dari transaksi terakhir memiliki variasi yang sangat tinggi, yang dapat menjadi indikator penting untuk mendeteksi anomali.
- Rasio Harga Pembelian: Variasi besar dalam rasio harga pembelian terhadap median menunjukkan bahwa transaksi dengan rasio yang sangat tinggi atau rendah mungkin mencurigakan.
- Transaksi Biner: Sebagian besar transaksi terjadi dengan pengecer yang sama, banyak yang menggunakan chip, tetapi sedikit yang menggunakan PIN.
- Transaksi Online: Proporsi yang signifikan dari transaksi adalah pesanan online, yang mungkin lebih rentan terhadap penipuan.
- Kelas Fraud: Hanya sekitar 8.74% dari transaksi yang merupakan fraud, menunjukkan ketidakseimbangan kelas yang perlu diperhatikan dalam pemodelan machine learning.
 
### _Exploratory Data Analysis_ (EDA)
#### _Univariate EDA_
##### Categorical Features
![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/univariate_repeat_reatiler.png?raw=true)
| nilai | jumlah sampel | persentase |
|-------|---------------|------------|
| 1.0   | 881536        | 88.2%      |
| 0.0   | 118464        | 11.8%      |

Terdapat 88.2 persen transaksi yang terjadi dari pengecer yang sama dan 11.8 % dari pengecer yang berbeda

![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/univariate_used_chip.png?raw=true)
| nilai | jumlah sampel | persentase |
|-------|---------------|------------|
| 0.0   | 649601        | 65.0%      |
| 1.0   | 350399        | 35.0%      |
Terdapat 35% transaksi yang mengggunakan chip (kartu kredit) dan 65 % tidak menggunakan kartu kredit

![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/univariate_used_pin_number.png?raw=true)
| nilai | jumlah sampel | persentase |
|-------|---------------|------------|
| 0.0   | 899392        | 89.9%      |
| 1.0   | 100608        | 10.1%      |
Terdapat 10.1% transaksi yang menggunakan nomor pin  dan 89.9% tidak menggunakan pin

![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/univariate_online_order.png?raw=true)
| nilai | jumlah sampel | persentase |
|-------|---------------|------------|
| 1.0   | 650552        | 65.1%      |
| 0.0   | 349448        | 34.9%      |

Terdapat 65.1% transaksi tersebut merupakan pesanan _online_ dan 34.9% merupakan pesanan _offline_ (pembayaran ditempat)

![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/univariate_fraud.png?raw=true)

| nilai | jumlah sampel | persentase |
|-------|---------------|------------|
| 0.0   | 912597        | 91.3%      |
| 1.0   | 87403         | 8.7%       |
Dari seluruh data transaksi yang terdapat 8.7% transaksi yang merupakan fraud (penipuan) dan yang bukan bukan merupakan penipuan 91.3%. Ini menandakan bahwa data mengalami ketidakseimbangan data.

##### _Numerical Features_
![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/univariate_numerical_features.png?raw=true)
1. _**Distance from Home**_
Distribusi: Sebagian besar transaksi terjadi dekat dengan rumah pemilik kartu.
Outliers: Beberapa transaksi terjadi sangat jauh dari rumah, hingga 10,000 km, yang mungkin menunjukkan potensi penipuan.

2. _**Distance from Last Transaction**_
Distribusi: Sebagian besar transaksi berikutnya terjadi dekat dengan lokasi transaksi sebelumnya.
Outliers: Ada beberapa jarak yang sangat besar, hingga 12,000 km, yang dapat mengindikasikan transaksi mencurigakan

3. _**Ratio to Median Purchase Price**_
Distribusi: Mayoritas transaksi berada di sekitar harga beli rata-rata.
Outliers: Beberapa transaksi memiliki rasio harga yang sangat tinggi, hingga lebih dari 250 kali harga rata-rata, yang mungkin merupakan indikasi penipuan.

#### _Multivariate EDA_
##### _Categorical Features_
![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/multivariate_categorical_features.png?raw=true)
**_Fraud_ vs. _Non-Fraud Transactions by Repeat Retailer_** 
Grafik ini menunjukkan hubungan antara transaksi yang dilakukan di pengecer yang sama dengan kejadian penipuan. Sebagian besar transaksi dilakukan di pengecer yang sama (repeat_retailer = 1.0) baik itu penipuan maupun bukan. Namun, terlihat bahwa penipuan lebih sedikit terjadi pada transaksi yang dilakukan di pengecer yang sama dibandingkan dengan transaksi yang dilakukan di pengecer yang berbeda.

**_Fraud vs. Non-Fraud Transactions by Used Chip_**
Grafik ini menunjukkan hubungan antara penggunaan chip dalam transaksi dengan kejadian penipuan. Terlihat bahwa lebih banyak transaksi tidak menggunakan chip (used_chip = 0.0) dibandingkan dengan yang menggunakan chip (used_chip = 1.0). Namun, transaksi tanpa chip memiliki jumlah penipuan yang lebih tinggi dibandingkan dengan transaksi yang menggunakan chip.

**_Fraud vs. Non-Fraud Transactions by Used Pin Number_**
Grafik ini menunjukkan hubungan antara penggunaan nomor PIN dalam transaksi dengan kejadian penipuan. Sebagian besar transaksi tidak menggunakan PIN (used_pin_number = 0.0). Dari data tersebut, terlihat bahwa transaksi yang tidak menggunakan PIN memiliki jumlah penipuan yang lebih tinggi dibandingkan dengan transaksi yang menggunakan PIN (used_pin_number = 1.0).

**_Fraud vs. Non-Fraud Transactions by Online Order_**
Grafik ini menunjukkan hubungan antara pesanan online dengan kejadian penipuan. Terlihat bahwa sebagian besar transaksi bukan pesanan online (online_order = 0.0). Dari data tersebut, terlihat bahwa penipuan lebih banyak terjadi pada transaksi yang bukan pesanan online dibandingkan dengan transaksi yang merupakan pesanan online.

##### _Numerical Features_
![Univariate repeat_retailer](https://github.com/stevanussmbrng/Credit-Card-Fraud-Detection-/blob/main/gambar/multivariate_numerical_feature.png?raw=true)
Fitur seperti ‘distance_from_last_transaction’, ‘ratio_to_median_purchase_price’, dan ‘fraud’ memiliki korelasi yang lebih tinggi, yang bisa menjadi indikator penting dalam mendeteksi transaksi fraud.

## Data Preparation

1. **Normalisasi Data** 
Variabel `distance_from_home`, `distance_from_last_transaction`, dan `ratio_to_median_purchase_price` ini memiliki rentang nilai yang berbeda sehingga perlu dilakukaan _standardization_. Standardization yang dilakukan menggunakan metode _Z-Score Standardization_ . Z-score dihitung dengan mengambil selisih antara angka dan mean (rata-rata) lalu membagi selisih yang diperoleh dengan standar deviasi [(“Data Analytical Tests,” 2014)](https://doi.org/10.1002/9781118936764.ch5).
2. **Pembagian Dataset**
Memisahkan data menjadi set pelatihan dan set pengujian untuk mengukur performa model secara objektif. Pembagian data dibagi dengan proporsi _data training_ 70% dan _data test_ 30%.
3. **Penanganan Data Tidak Seimbang**
- Tidak seimbangan data dapat mempengaruhi hasil akhir dalam memperoleh nilai akurasi [(Febriady dkk., 2022)](https://doi.org/10.30865/mib.v6i2.3515). Dengan menggunakan data yang tidak seimbang, model machine learning yang dibangun akan lebih condong untuk mempelajari pola kelas mayoritas dan mengabaikan kelas minoritas. Hal tersebut mengakibatkan model menjadi lemah dan menghasilkan performa yang buruk [(Ningsih dkk., 2022)](https://doi.org/10.37012/jtik.v8i2.1306). 
- Menggunakan teknik seperti SMOTENC (_Synthetic Minority Over-sampling Technique for Nominal and Continuous_) untuk menangani ketidakseimbangan kelas dalam dataset. SMOTENC adalah metode oversampling yang paling umum digunakan, yang bertujuan untuk meningkatkan kelas minoritas dengan mereplikasi kelas minoritas yang ada. Teknologi ini digunakan dengan N, yang mengindikasikan bahwa set data yang diusulkan berisi data nominal. Teknologi ini menciptakan sintesis antara kelas minoritas yang ada dan kelas minoritas baru di area yang lebih dekat dengan kelas minoritas yang ada dengan menggunakan metode klasifikasi _K-Nearest Neighbourhood_ (KNN). Metode ini menghasilkan pelatihan virtual dari kelas minoritas yang baru dibuat dengan menginterpolasi kelas minoritas yang ada. Catatan pelatihan sintetis dipilih secara acak menggunakan proses klasifikasi KNN, dan data direkonstruksi setelah _oversampling_ [(Ravindranath dkk., 2024)](https://doi.org/10.1016/j.asoc.2024.111698).
    ##### Sebelum Resampling
    | Fraud | Jumlah Sampel |
    |-------|---------------|
    | 0.0   | 638818        |
    | 1.0   | 61182         |

    ##### Setelah Resampling
    | Fraud | Jumlah Sampel |
    |-------|---------------|
    | 0.0   | 638818        |
    | 1.0   | 638818        |


## Modeling
Pada tahapan modeling ini, mencoba untuk membanding 3 model klasifikasi dengan menggunakan parameter default dan hanya menginisiasi `randomstate = 42`. Hal ini dilakukan karena hasil dari modeling sudah memberikan hasil yang baik untuk dataset ini.
1.  **Logistic Regression**
_Logistic Regression_ adalah metode statistika untuk menganalisis hubungan antara variabel respon (Y) yang bersifat kategorikal dengan satu atau lebih variabel prediktor (X)1. Metode ini digunakan ketika variabel respon memiliki dua kategori (biner logistik) atau lebih (ordinal logistik), seperti dalam kasus prediksi kebangkrutan perusahaan, penilaian akreditasi sekolah, atau analisis stres mahasiswa. Regresi Logistik sering dipilih ketika asumsi regresi linier tidak terpenuhi, misalnya ketika variabel respon tidak terdistribusi normal atau terjadi heteroskedastisitas. Dalam konteks yang berbeda, penelitian ini menunjukkan aplikasi Regresi Logistik dalam berbagai bidang, seperti pendidikan, manajemen keuangan, dan psikologi, menekankan pentingnya metode ini dalam mengatasi masalah klasifikasi dan prediksi dalam data kategorikal.[(Suliadi, 2015)](https://www.academia.edu/68466086/Regresi_Logistik_Pada_Data_Rare_Event)
2. **Random Forest**
_Random Forest_ adalah pengembangan dari metode Decision Tree yang memanfaatkan beberapa Decision Tree untuk membuat keputusan. Random Forest bekerja dengan cara menggabungkan hasil dari beberapa pohon keputusan yang dilatih pada subset fitur acak. Keputusan akhir diambil berdasarkan hasil dari semua pohon keputusan dalam ensemble. Random Forest efisien untuk dataset besar dan dapat meningkatkan performa model klasifikasi. Dengan adanya seleksi fitur tentu Random Forest dapat bekerja pada big data dengan parameter yang kompleks secara efektif [(Devella dkk., 2020)](https://doi.org/10.35957/jatisi.v7i2.289).
3. **XGBoost**: 
XGBoost adalah sistem tree boosting yang sangat efektif dan banyak digunakan dalam machine learning. Dalam jurnal berjudul “XGBoost: A Scalable Tree Boosting System,” peneliti menggambarkan XGBoost sebagai sistem tree boosting yang dapat diukur dari awal hingga akhir, dan banyak digunakan oleh data scientist untuk mencapai hasil terbaik pada banyak tantangan _machine learning_. Mereka mengusulkan algoritma yang memperhatikan sparsity untuk data yang jarang dan _weighted quantile sketch_ untuk pembelajaran pohon yang mendekati. Selain itu, penelitian ini memberikan wawasan tentang pola akses cache, kompresi data, dan sharding untuk membangun sistem tree boosting yang dapat diukur. XGBoost mampu mengatasi miliaran contoh dengan menggunakan sumber daya yang lebih sedikit daripada sistem yang ada [(Tianqi Chen, 2016)](https://arxiv.org/abs/1603.02754).

## Evaluation

1. **Logistic Regression**:
- Confusion Matrix
    |             | Predicted non-fraud | Predicted fraud |
    |-------------|--------------|--------------|
    | Non-fraud  | 259071       | 14708        |
    | Fraud  | 1868         | 24353        |
- Classification Report
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | Non-fraud   | 0.99      | 0.95   | 0.97     | 273779  |
    | Fraud  | 0.62      | 0.93   | 0.75     | 26221   |
    |       |           |        |          |         |
    | **Accuracy**  |       |        | 0.94     | 300000 |
    | **Macro Avg** | 0.81      | 0.94   | 0.86     | 300000 |
    | **Weighted Avg** | 0.96      | 0.94   | 0.95     | 300000 |
- Hasil Evaluasi
    | Metric    | Value  |
    |-----------|--------|
    | Accuracy  | 94.47% |
    | Precision | 62.32% |
    | Recall    | 92.87% |
    | F1-Score  | 74.66% |

2. **Random Forest**
- Confusion Matrix
    |             | Predicted non-fraud | Predicted fraud |
    |-------------|--------------|--------------|
    | Non-fraud   | 273079       | 700          |
    | Fraud  | 7            | 26214        |
- Classification Report
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | Non-Fraud  | 1.00      | 1.00   | 1.00     | 273779  |
    | Fraud   | 0.97      | 1.00   | 0.99     | 26221   |
    |       |           |        |          |         |
    | **Accuracy**  |       |        | 1.00     | 300000 |
    | **Macro Avg** | 0.99      | 1.00   | 0.99     | 300000 |
    | **Weighted Avg** | 1.00      | 1.00   | 1.00     | 300000 |
- Hasil Evaluasi
    | Metric    | Value  |
    |-----------|--------|
    | Accuracy  | 99.76% |
    | Precision | 97.48% |
    | Recall    | 99.97% |
    | F1-Score  | 98.72% |

3. **XGBoost**
- Confusion Matrix
    |             | Predicted non-fraud | Predicted fraud |
    |-------------|--------------|--------------|
    | Non-fraud  | 272773       | 1006         |
    | Fraud  | 167          | 26054        |
- Classification Report
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|---------|
    | Non-fraud   | 1.00      | 1.00   | 1.00     | 273779  |
    | Fraud   | 0.96      | 0.99   | 0.98     | 26221   |
    |       |           |        |          |         |
    | **Accuracy**  |       |        | 1.00     | 300000 |
    | **Macro Avg** | 0.98      | 0.99   | 0.99     | 300000 |
    | **Weighted Avg** | 1.00      | 1.00   | 1.00     | 300000 |
- Hasil Evaluasi
    | Metric    | Value  |
    |-----------|--------|
    | Accuracy  | 99.61% |
    | Precision | 96.14% |
    | Recall    | 99.36% |
    | F1-Score  | 97.72% |

Penjelasan:
- Logistic Regression: Logistic Regression memiliki akurasi yang tinggi (94.47%) dan performa yang baik dalam mendeteksi transaksi non-fraud (precision 0.99, recall 0.95). Namun, performa dalam mendeteksi transaksi fraud lebih rendah dengan precision 0.62 dan recall 0.93. Ini berarti bahwa model ini cenderung memberikan banyak false positives tetapi lebih jarang mengabaikan kasus fraud.
- Random Forest: Random Forest menunjukkan performa yang sangat baik dengan akurasi 99.76%. Model ini hampir sempurna dalam mendeteksi kedua kelas dengan precision, recall, dan f1-score yang tinggi. Ini berarti bahwa model ini sangat efektif dalam membedakan antara transaksi fraud dan non-fraud dengan kesalahan yang sangat kecil.
- XGBoost: XGBoost juga menunjukkan performa yang sangat baik dengan akurasi 99.61%. Precision dan recall untuk mendeteksi transaksi fraud sedikit lebih rendah dibandingkan Random Forest, tetapi tetap sangat tinggi. Ini berarti bahwa model ini juga sangat efektif dalam mendeteksi transaksi fraud dan non-fraud, dengan sedikit lebih banyak kesalahan dibandingkan Random Forest.

Hasil evaluasi menunjukkan bahwa model **Random Forest** memberikan performa terbaik dengan Accuracy, Precision, Recall, dan F1-Score yang paling tinggi diantara model lainnya. Model **XGBoost** juga memiliki performa sedikit di bawah Random Forest dalam mendeteksi fraud, tetapi masih sangat efektif.
