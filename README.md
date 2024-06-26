# Laporan Proyek _Machine Learning_ - Stevanus Sembiring

## DETEKSI PENIPUAN KARTU KREDIT MENGGUNAKAN _SUPERVISED MACHINE LEARNING_

Masifnya penggunaan teknologi saat ini, didorong adanya transformasi digital pasca terjadinya pandemi COVID-19, tidak terlepas mendorong perubahan pada pola pembayaran yang awalnya dilakukan secara tunai menjadi non-tunai (_cash-less_). Hingga kini, penggunaan mode pembayaran menggunakan kartu kredit bertambah seiring waktu di mana penggunaan kartu kredit menjadi pilihan karena tidak harus melakukan pembayaran secara tunai. Berdasarkan statistik Sistem Pembayaran Dan Infrastruktur Pasar Keuangan (SPIP) (Bank Indonesia, 2024) jumlah pemegang kartu kredit di Indonesia sebanyak 17.727.481 unit sedangkan volume transaksinya sebesar 36.609.893 transaksi, kemudian nilai transaksi yang bersumber dari kartu kredit sebanyak Rp37,918 Miliar.

Dibuktikan dengan banyaknya pengguna dari informasi statistik Bank Indonesia, menjadikan salah satu faktor yang menjadikan kartu kredit sebagai target utama tindak pidana seperti penipuan (_fraud_) yakni transaksi tidak sah yang dilakukan oleh orang yang tidak dikenali dengan memanfaatkan adanya kebocoran data pribadi pengguna. Keamanan dari data pribadi memang menjadi tanggung jawab bersama antara pihak penyedia layanan kartu kredit (bank) dengan nasabah yang memegang kartu kredit. Sehingga pihak yang memiliki kepentingan wajib dengan sungguh-sungguh melakukan penjagaan agar data pribadi dari kartu kredit tidak dapat disalahgunakan oleh orang yang tidak bertanggung jawab. Hingga saat ini telah dilakukan berbagai cara lain untuk dapat melakukan tindakan preventif dengan melakukan otomasi agar dapat dilakukan proses pendeteksian secara sedini mungkin.

Dari banyaknya jumlah, volume dan nilai transaksi yang terjadi pada proses kartu kredit secara terus-menerus dari waktu ke waktu, maka solusi untuk dapat menangani masalah ini yakni mengembangkan sistem pendeteksian penipuan (_fraud_) yang dapat secara cepat, otomatis akurat, serta bekerja terus-menerus yakni dengan membangun model supervised machine learning. Berdasarkan penelitian yang dilakukan oleh (Ningsih dkk., 2022) dilakukan pembuatan model machine learning menggunakan _Decision Tree_, _Random Forest_, _Logistic Regression_, dan SVM (_Support Vector Machines_) menunjukkan bahwa model Random Forest menghasilkan nilai performa keseluruhan paling baik dibandingkan algoritma lainnya, yaitu menghasilkan akurasi sebesar 92%. Penelitian lain juga menggunakan algoritma yang sama yaitu _Random Forest_ untuk dapat mendeteksi adanya _fraud_ pada kartu kredit yang dilakukan (Kurniawan & Yulianingsih, 2021) di mana model yang dihasilkan dapat menghasilkan akurasi sebesar 85%.

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

![Univariate repeat_retailer](https://drive.google.com/uc?export=view&id=1abcdEFGhij)

## Data Preparation

Pada bagian ini, teknik data preparation yang dilakukan akan dijelaskan.

- **Normalisasi Data**: Menggunakan teknik standardisasi untuk mengubah skala variabel `distance_from_home`, `distance_from_last_transaction`, dan `ratio_to_median_purchase_price` karena variabel ini memiliki rentang nilai yang berbeda.
- **Pembagian Dataset**: Memisahkan data menjadi set pelatihan dan set pengujian untuk mengukur performa model secara objektif.
- **Penanganan Data Tidak Seimbang**: Menggunakan teknik seperti SMOTE (Synthetic Minority Over-sampling Technique) untuk menangani ketidakseimbangan kelas dalam dataset.

**Rubrik/Kriteria Tambahan**: 
- Normalisasi dan penanganan data tidak seimbang sangat penting untuk meningkatkan performa model machine learning. Normalisasi memastikan bahwa semua fitur berada dalam skala yang konsisten, sehingga algoritma lebih cepat konvergen dan stabil secara numerik. Ini penting karena fitur dengan skala yang berbeda dapat mendominasi perhitungan dan menyebabkan bias dalam model. Sementara itu, penanganan data tidak seimbang diperlukan karena model cenderung bias terhadap kelas mayoritas dalam dataset yang tidak seimbang. Dengan teknik seperti oversampling atau undersampling, model dapat belajar lebih baik dari kelas minoritas, menghasilkan prediksi yang lebih akurat dan adil. Teknik ini juga meningkatkan metrik evaluasi penting seperti precision, recall, dan F1-score, yang memberikan gambaran lebih baik tentang performa model dalam kasus dengan data tidak seimbang.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan.

- **Logistic Regression**: Algoritma dasar yang sering digunakan untuk klasifikasi biner.
- **Random Forest**: Algoritma ensemble yang dapat menangani dataset dengan baik dan memiliki kemampuan untuk mengurangi overfitting.
- **XGBoost**: Algoritma ini daapt mengurangi overfitting dengan menggunakan teknik regularization dan pruning.

**Rubrik/Kriteria Tambahan**: 
1. **Logistic Regression**:
- Kelebihan:
    - Mudah diinterpretasi dan diimplementasi.
    - Cocok untuk masalah dengan ruang fitur yang relatif sederhana.
    - Efisien dalam waktu komputasi dan memori.
- Kekurangan:
    - Biasanya tidak menangani relasi non-linear antara fitur dan target..
    - Rentan terhadap overfitting jika tidak ada penanganan khusus terhadap fitur-fitur yang tidak relevan atau redundan.
- Hyperparameter Tuning: 
    - Beberapa hyperparameter yang penting untuk disesuaikan adalah regularisasi (penalty), kekuatan regularisasi (C), dan jenis solver (misalnya, 'liblinear', 'lbfgs').
2.  **Random Forest**:
- Kelebihan:
    - Mampu menangani hubungan non-linear dan interaksi antar fitur dengan baik.
    - Tidak memerlukan normalisasi data.
    - Cenderung tidak overfitting karena menggunakan multiple decision trees.
- Kekurangan:
    - Lebih kompleks secara komputasi dibandingkan dengan Logistic Regression.
    - Sulit untuk diinterpretasi karena merupakan gabungan dari banyak decision trees.
- Hyperparameter Tuning: 
    - Hyperparameter yang perlu disesuaikan meliputi jumlah pohon (n_estimators)kedalaman maksimal setiap pohon (max_depth), jumlah fitur yang dipertimbangkan untuk splitting (max_features), dll.
3. **XGBoost**
- Kelebihan:
    - Performa yang sangat baik karena menggunakan teknik boosting.
    - Mampu menangani dataset besar dengan fitur-fitur yang kompleks.
    - Memberikan feature importance yang bisa diinterpretasi.
- Kekurangan:
    - Rentan terhadap overfitting jika hyperparameter tidak diatur dengan baik.
    - Memerlukan waktu yang lebih lama untuk training dibandingkan dengan Random Forest.
- Hyperparameter Tuning: 
    - Hyperparameter yang penting antara lain learning rate (eta), jumlah iterasi (num_boost_round), kedalaman maksimal pohon (max_depth), dan lambda (regularization term).

Model terbaik dipilih berdasarkan metrik evaluasi yang paling sesuai dengan kebutuhan bisnis untuk deteksi fraud. Misalnya, jika prioritas adalah mendeteksi sebanyak mungkin fraud (tinggi pada recall), maka model dengan recall tertinggi akan dipilih. Jika biaya false positive menjadi perhatian (tinggi pada precision), maka model dengan precision yang tinggi akan lebih baik.

## Evaluation

1. **Logistic Regression**:
- Hasil Evaluasi:
Accuracy: 93.45%
Precision (kelas 0): 0.99
Precision (kelas 1): 0.58
Recall (kelas 0): 0.93
Recall (kelas 1): 0.95
F1-score (kelas 0): 0.96
F1-score (kelas 1): 0.72
- Confusion Matrix:
[[255473  18306]
 [  1351  24870]]
2. **Random Forest**
- Hasil Evaluasi:
Accuracy: 100.00%
Precision dan Recall (kedua kelas): 1.00
F1-score (kedua kelas): 1.00
- Confusion Matrix:
[[273779      0]
 [     4  26217]]
3. **XGBoost**
- Hasil Evaluasi:
Accuracy: 99.84%
Precision (kelas 0): 1.00
Precision (kelas 1): 0.99
Recall (kelas 0): 1.00
Recall (kelas 1): 1.00
F1-score (kelas 0): 1.00
F1-score (kelas 1): 0.99
- Confusion Matrix:
[[273396    383]
 [   102  26119]]

Hasil evaluasi menunjukkan bahwa model **Random Forest** memberikan performa terbaik dengan Accuracy, Precision, Recall, dan F1-Score 100%.

**Rubrik/Kriteria Tambahan**: 
- Random Forest: Model ini menunjukkan performa sempurna dengan akurasi 100% dan metrik evaluasi lainnya juga sempurna. Namun, performa yang sempurna ini bisa jadi indikasi overfitting, terutama jika dataset Anda besar dan kompleks.
- XGBoost: Meskipun tidak sempurna seperti Random Forest, XGBoost menunjukkan performa yang sangat tinggi dengan akurasi 99.84%, precision dan recall mendekati sempurna, dan F1-score yang hampir sempurna. XGBoost cenderung lebih robust dan generalizable dibandingkan dengan Random Forest karena kemampuannya dalam menangani dataset yang lebih besar dan lebih kompleks.


