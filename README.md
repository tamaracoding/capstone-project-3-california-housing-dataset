# <div style="padding: 35px;color:#FF5733;margin:10;font-size:60%;text-align:center;display:fill;border-radius:10px;border: 2px solid black;background-color:transparent;overflow:hidden;background-color:transparent"><b><span style='color:#FFFFFF'></span></b> <b>EKSPERIMEN MODEL REGRESI UNTUK PREDIKSI HARGA PROPERTI: PENGARUH MULTIKOLINEARITAS TERHADAP MODEL PARAMETRIK VS NON PARAMETRIK</b></div>

![alur modeling drawio](https://github.com/user-attachments/assets/b88195f9-7bb2-4b88-8d20-28926a023d4d)

<div style="color:white;display:fill;border-radius:8px;font-size:100%; letter-spacing:1.0px;"><p style="padding: 5px;color:white;text-align:left;"><b><span style='color:#FF5733'>1.1 LATAR BELAKANG MASALAH</span></b></p></div>

Banyak faktor yang mempengaruhi nilai properti rumah, seperti lokasi, ukuran bangunan, kondisi lingkungan sekitar, serta usia bangunan. Faktor-faktor ini sangat bervariasi antar properti. Pasar properti sendiri merupakan industri yang sangat fluktuatif, yang sangat bergantung pada perubahan permintaan dan penawaran, serta faktor ekonomi seperti suku bunga dan inflasi. Oleh karena itu, **memprediksi variasi harga properti seiring waktu adalah tantangan yang cukup besar.**

Keterbatasan data juga menjadi tantangan tersendiri dalam prediksi. Data properti dunia nyata sering kali memiliki fitur-fitur yang terbatas, akibatnya  prediksi harga properti sering kali tidak dapat memperhitungkan semua variabel yang relevan. Sehingga proses rekayasa fitur (_feature engineering_) sangat krusial untuk meningkatkan akurasi prediksi.

Selain itu, salah satu masalah yang sering muncul dalam prediksi harga properti adalah multikolinearitas. Multikolinearitas terjadi ketika dua atau lebih variabel prediktor dalam dataset memiliki korelasi tinggi satu sama lain, sehingga saling tumpang tindih dalam menjelaskan variasi target (harga properti). Ini bisa menjadi masalah khususnya dalam model parametrik, seperti regresi linier, karena akan membuat estimasi koefisien menjadi tidak stabil dan sulit diinterpretasikan.

Pada model parametrik, multikolinearitas dapat menyebabkan kenaikan varians dari koefisien, sehingga mengurangi kemampuan model untuk memprediksi secara akurat. Oleh karena itu, kita perlu mengidentifikasi dan menangani multikolinearitas dengan teknik seperti Variance Inflation Factor (VIF) atau dengan melakukan regularisasi pada model (Contoh: Ridge Regression).

Namun, dalam model non-parametrik seperti Random Forest dan XGBoost, multikolinearitas tidak menimbulkan masalah yang signifikan. Ini karena model ensemble berbasis tree memilih subset fitur secara acak untuk setiap pohon, sehingga dampak korelasi antar fitur tidak terlalu kuat dibandingkan model linier. [(Idaho State University)](https://ar5iv.labs.arxiv.org/html/2111.02513)

<div style="color:white;display:fill;border-radius:8px;font-size:100%; letter-spacing:1.0px;"><p style="padding: 5px;color:white;text-align:left;"><b><span style='color:#FF5733'>1.2 PENJELASAN DATASET YANG DIGUNAKAN</span></b></p></div>

`California housing dataset` berisi atribut terkait rumah untuk properti yang berlokasi di California, yang akan digunakan dalam studi ini. Dataset ini diambil dari bab kedua buku karya Aurélien Géron yang berjudul `Hands-On Machine Learning with Scikit-Learn and TensorFlow`. Dataset ini berisi informasi dari sensus California tahun 1990 dan merupakan buku pengantar yang sangat direkomendasikan untuk menerapkan algoritma pembelajaran mesin (_machine learning_). Dataset ini mencakup berbagai fitur/prediktor yang terkait dengan perumahan dan demografi di California, sehingga cocok untuk pembelajaran dasar-dasar machine learning. [(amazon.com)](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646).

**Berikut ini adalah kolom-kolom pada dataset California Housing:**

`Variabel Target:`
| Target           | Deskripsi Target                                                                                              |
|--------------------|-----------------------------------------------------------------------------------------------------------|
| median_house_value | Nilai median rumah rata-rata untuk lingkungan rumah tangga dalam satu blok (diukur dalam satuan Dolar AS) |

`Variabel Predictor/Features:`
| Fitur/Prediktor    | Desksripsi Fitur/Prediktor                                                                                             |
|--------------------|-----------------------------------------------------------------------------------------------------------|
| longitude          | Letak rumah berdasarkan koordinat geografi (semakin tinggi angkanya, semakin ke barat).                    |
| latitude           | Letak rumah berdasarkan koordinat geografi (semakin tinggi angkanya, semakin ke utara).                    |
| housing_median_age | Median usia rumah; Semakin rendah angkanya, menandakan usia bangunan rumah yang masih muda, dan vice versa |
| total_rooms        | Jumlah ruangan pada rumah-rumah dalam suatu blok perumahan                                                 |
| total_bedrooms     | Jumlah kamar tidur pada rumah-rumah dalam suatu blok perumahan                                             |
| population         | Jumlah total orang yang tinggal di dalam satu blok perumahan                                               |
| households         | Jumlah total rumah tangga, sekelompok orang yang tinggal dalam satu unit rumah, untuk satu blok perumahan   |
| median_income      | Pendapatan rata-rata rumah tangga dalam satu blok rumah (diukur dalam puluhan ribu Dolar AS)               |
| ocean_proximity    | Kedekatan lokasi dari area pesisir/ pantai/ laut                                                           |

<div style="color:white;display:fill;border-radius:8px;font-size:100%; letter-spacing:1.0px;"><p style="padding: 5px;color:white;text-align:left;"><b><span style='color:#FF5733'>1.3 TUJUAN</span></b></p></div>

`Tujuan umum `

Memenuhi kewajiban penyusunan `Capstone Project Modul 3: Program Data Science and Machine Learning Purwadhika Digital School`. 

`Tujuan khusus`
- Membangun model machine learning untuk memprediksi harga properti di Californial dengan akurasi yang optimal
- Memahami dan mendemonstrasikan bagaimana model-model sederhana hingga model yang lebih kompleks (seperti ensemble model) bekerja dengan memanfaatkan pustaka `scikit-learn`. Fokusnya bukan pada kesempurnaan model saja, tetapi juga pada pemahaman mendalam tentang proses pembuatan model dan bagaimana setiap jenis model bekerja secara teknis dan konseptual.
- Membandingkan performa berbagai model (sederhana hingga kompleks) untuk mengidentifikasi model mana yang paling efisien dalam hal kompleksitas, akurasi, dan kecepatan.
- Memahami teknik pengolahan dataset terbaik untuk model regresi, melalui proses pembersihan data, penanganan outliers, mengatasi multikolinearitas, serta melakukan rekayasa fitur (_feature engineering_) seperti normalisasi, scaling, encoding, dan pembuatan fitur baru.
- Melakukan evaluasi model dengan berbagai metrik, `seperti mean squared error` (MSE), `root mean squared error` (RMSE), dan `R-squared`, untuk memahami performa model dalam memprediksi variabel target.
- Menerapkan teknik validasi silang (`cross-validation`) guna memastikan bahwa model tidak overfitting dan mampu melakukan generalisasi dengan baik pada data baru.
- Menilai dampak Bayesian Optimization Hyperparameter Tuning pada performa model non-parametrik seperti Random Forest dan XGBoostRegresor, serta mengevaluasi peningkatan kinerja model setelah dilakukan tuning.
- Menghasilkan insight dari model yang dibangun mengenai faktor-faktor utama yang mempengaruhi harga rumah di California, untuk memahami bagaimana berbagai fitur mempengaruhi nilai properti.

<div style="color:white;display:fill;border-radius:8px;font-size:100%; letter-spacing:1.0px;"><p style="padding: 5px;color:white;text-align:left;"><b><span style='color:#FF5733'>1.4 MODEL YANG DIGUNAKAN</span></b></p></div>

- Rule-Based Model: Model sederhana yang saya kustomisasi sebagai pembanding untuk evaluasi performa.

**Model Linear Parametrik:**
- Linear Regression
- Ridge Regression

**Model Non-Parametrik:**
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extreme Gradient Boosting
