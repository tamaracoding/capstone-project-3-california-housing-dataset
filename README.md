# <div style="padding: 35px;color:#FF5733;margin:10;font-size:60%;text-align:center;display:fill;border-radius:10px;border: 2px solid black;background-color:transparent;overflow:hidden;background-color:transparent"><b><span style='color:#FFFFFF'></span></b> <b>EKSPERIMEN MODEL REGRESI UNTUK PREDIKSI HARGA PROPERTI: PENGARUH MULTIKOLINEARITAS TERHADAP MODEL PARAMETRIK VS NON PARAMETRIK</b></div>

**Note:** link notebook yang memiliki map interaktif==> https://nbviewer.org/github/tamaracoding/capstone-project-3-california-housing-dataset/blob/main/CAPSTONE%20PROJECT%203%20-%20REGRESI.ipynb

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

## Kesimpulan

- Model terbaik untuk dataset California Housing adalah XGBRegressor yang telah melalui Hyperparameter-Tuning serta fitur-fiturnya telah direduksi (direduksi dari dataset yang sudah melalui _feature engineering_, bukan sebelum). Kombinasi dari tuning dan reduksi fitur ini berhasil memaksimalkan performa model XGBoost. Bagi pihak-pihak yang ingin memaksimalkan akurasi prediksi harga properti di California, model ini adalah pilihan terbaik yang menggabungkan ketepatan prediksi dan kekuatan generalisasi.

- Rule-Based model adalah model yang dibangun dengan aturan sederhana "_if-then_" atau "_if-else_". Model rule-based berperan sebagai baseline model.Performa Rule-Based model meningkat signifikan setelah outliers pada data awal (df) dihapus, RMSE turun dari 92293.268 menjadi 74306.075, sehingga outliers memiliki dampak negatif dalam dataset ini.

- RMSE pada Rule-Based model tanpa outliers sebesar 74,306.075, sementara model terbaik menunjukkan penurunan drastis dalam RMSE menjadi 38,959.215. Dengan penurunan ini,  <span style="background-color: salmon">berbagai pihak yang menggunakan model ini dapat menekan kesalahan prediksi sebesar 35,346.86. Artinya, jika setiap unit kesalahan prediksi berkorelasi langsung dengan kerugian finansial, model terbaik ini mampu mengurangi potensi kerugian hampir 47.6% dibandingkan dengan model rule-based biasa yang hanya dihilangkan outliers-nya.</span>

- Dataset yang memiliki multikolinearitas (skor vif tinggi) justru meningkatkan performa seluruh model (RMSE turun pada kisaran USD 1.000 hingga 2.900), dibandingkan dengan dataset yang memiliki skor vif < 10. Bahkan dataset tersebut juga meningkatkan performa model linear yang dianggap sensitif terhadap multikolinearitas.

- Penambahan fitur-fitur baru pada model—meskipun secara teknis meningkatkan multikolinearitas—nyatanya justru dapat meningkatkan performa model, terutama model ensemble. Model ensemble seperti XGBoost atau Random Forest memiliki mekanisme internal yang lebih robust terhadap multikolinearitas, sehingga dapat tetap memanfaatkan informasi dari fitur tambahan tanpa terpengaruh secara signifikan oleh korelasi antar fitur.

- Fitur-fitur baru dapat membantu model ensemble untuk menangkap lebih banyak variasi dan pola yang ada dalam data, sehingga dapat meningkatkan akurasi prediksi. Pada model ensemble, penambahan fitur ini tidak menghasilkan overfitting yang sama seperti pada model regresi linier tradisional, karena model ensemble dapat mengatasi masalah redundansi informasi/kemiripan informasi melalui pengambilan keputusan kolektif dari banyak pohon keputusan yang digunakan dalam proses prediksi.

- Fitur-fitur bawaan dari dataset menunjukkan skor importance yang jauh lebih kecil jika dibandingkan dengan fitur-fitur baru yang dihasilkan dari transformasi atau rekayasa fitur (_feature engineering_) terhadap fitur bawaan tersebut.

- Hyperparameter tuning dengan Bayesian Optimization dapat dilakukan dengan sangat cepat (kurang dari 10 menit, bahkan bisa kurang dari 5 menit tergantung jumlah iterasi yang diinginkan). Performa model yang sudah dituning dengan metode ini, meningkat dengan signifikan dalam seluruh metrik evaluasi (RMSE, MAE, R-Squared, bahkan Standar Deviasi dari masing-masing fold).

## Limitasi Model

- Model ini hanya didasarkan pada data harga rumah di wilayah California dan tidak dapat digunakan secara langsung untuk wilayah lain tanpa adanya data tambahan yang relevan.

- Model ini cenderung kurang akurat untuk memprediksi harga rumah yang jauh dari rentang harga umum, terutama untuk rumah mewah atau rumah dengan harga jauh di bawah rata-rata pasar.

- Untuk prediksi rumah di California sendiri, <span style="background-color: salmon">model ini akan menghasilkan prediksi terbaik untuk rumah yang berada dalam rentang harga median di bawah USD 300.000</span>

- Model ini tidak mempertimbangkan faktor eksternal seperti tingkat kriminalitas, kondisi fasilitas umum, pajak bumi bangunan, dll

- Model ini tidak mempertimbangkan faktor internal rumah tambahan seperti ada atau tidaknya perapian, kolam renang, kanopi, balkon, teras, jenis bahan bangunan yang digunakan, dll

- Model ini akan sangat cocok untuk dataset yang memiliki fitur/target seperti berikut ini:
 longitude, latitude, housing median age, median income, median house value, ocean proximity, room per household, room per capita, bedroom ratio, income per capita, age binned custom, distance to LA, dan distance to Silicon Valley.

## Rekomendasi

Untuk meningkatkan performa model di masa mendatang, beberapa hal yang dapat dipertimbangkan antara lain:

- Hanya dengan menghilangkan outliers, kita sudah bisa menekan kesalahan prediksi sebesar USD 17.987,193 (RMSE). Sehingga, melakukan filter outliers dengan metode yang lebih canggih daripada IQR, seperti: DBSCAN (Density-Based Spatial Clustering of Applications with Noise), Isolation Forest, Principal Component Analysis (PCA) untuk deteksi outliers, Z-Score, dan lain-lain, kemungkinan dapat mengurangi kesalahan prediksi lebih dari USD 17.987,193.

- Melakukan hyperparameter tuning dengan bayesian optimization dapat menurunkan kesalahan prediksi sebesar USD 471 (RMSE), sehingga melakukan lebih banyak iterasi dalam optimasi model menggunakan Bayesian Optimization dapat mengurangi kesalahan prediksi lebih dari USD 471. Dalam notebook ini, iterasi pencarian hanya dilakukan sebanyak 25 kali (sekitar 4 menit). 

- Kombinasi penggunaan model XGBoost Regressor + Filter Outliers + Hyperparameter Tuning (Bayesian Opt.) + Reduksi 5 fitur dengan koefisien importance terendah, dapat mengurangi kesalahan prediksi total sebesar USD 53.334,053. Sehingga, jika hyperparameter tuning bayesian optimization ditingkatkan iterasinya (misal >50) + Filter Outliers dilakukan dengan metode yang lebih canggih dibandingkan IQR + Inspeksi feature importance lebih menyeluruh, kesalahan prediksi total dapat dikurangi lebih dari USD 53.334,053.

- Dataset yang memiliki multikolinearitas (skor vif tinggi) justru meningkatkan performa seluruh model (RMSE turun pada kisaran USD 1.000 hingga 2.900), dibandingkan dengan dataset yang memiliki skor vif < 10. Sehingga, disarankan untuk menggunakan dataset dengan lebih banyak fitur, meskipun memiliki multikolinearitas. Terutama, untuk model ensembel yang tahan terhadap korelasi antar fitur.

- Menambahkan fitur jarak rumah ke fasilitas transportasi umum, bandara besar seperti LAX, serta kota-kota besar seperti San Francisco, San Diego, dan/atau San Jose. Fitur ini dapat membantu model menangkap variasi harga yang berkaitan dengan aksesibilitas lokasi.

- Menyertakan kolom fitur biner 'san_fransisco_prop' (yes/no), yang dibuat berdasarkan koordinat latitude dan longitude San Francisco. Mengingat San Francisco merupakan salah satu kota dengan harga properti tertinggi di dunia, fitur ini akan memberikan kontribusi signifikan terhadap prediksi model.

- Mengintegrasikan informasi eksternal terkait lingkungan perumahan, seperti tingkat kriminalitas, kondisi fasilitas umum, dan pajak properti. Data tambahan ini bisa membantu model lebih akurat dalam menangkap faktor eksternal yang mempengaruhi harga rumah.

- Menyertakan fitur-fitur rumah lainnya seperti jenis bahan bangunan, ada/tidaknya perapian, kolam renang, teras, halaman belakang (backyard), rubanah (basement), taman, balkon, dan lain-lain

- Membuat kategori khusus berdasarkan tipe properti (seperti rumah tapak, apartemen, atau townhouse) dapat membantu model lebih akurat dalam memprediksi harga karena setiap tipe properti mungkin memiliki pola harga yang berbeda.

- Selain RMSE dan MAE, pertimbangkan untuk menggunakan metrik evaluasi lain seperti _R-squared adjusted_ untuk mengevaluasi seberapa baik model mengakomodasi variabel prediktor, atau MSLE (_Mean Squared Logarithmic Error_) yang lebih sensitif terhadap kesalahan prediksi pada properti berharga rendah.


- Selain RMSE dan MAE, pertimbangkan untuk menggunakan metrik evaluasi lain seperti _R-squared adjusted_ untuk mengevaluasi seberapa baik model mengakomodasi variabel prediktor, atau MSLE (_Mean Squared Logarithmic Error_) yang lebih sensitif terhadap kesalahan prediksi pada properti berharga rendah.

- Melakukan filter outliers dengan metode yang lebih canggih daripada IQR, seperti: DBSCAN (Density-Based Spatial Clustering of Applications with Noise), Isolation Forest, Principal Component Analysis (PCA) untuk deteksi outliers, Z-Score, dan lain-lain

