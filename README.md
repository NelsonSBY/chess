# Laporan Proyek Machine Learning — Nelson Lau
## Domain Proyek
Domain yang dipilih untuk proyek machine learning terapan ini adalah Kesehatan, dengan judul: **Klasifikasi Rumah Sakit**

<img src="https://qtxasset.com/cdn-cgi/image/w=850,h=478,f=auto,fit=crop,g=0.5x0.5/https://qtxasset.com/quartz/qcloud5/media/image/Hospital%20beds.jpg?VersionId=2UcFqhix9C8xLqX.GsJlJZ.zJQ9NGMdn" alt="Ilustrasi Rumah Sakit" width="600"/>

### Latar Belakang

Sektor kesehatan merupakan pilar penting dalam pembangunan nasional. Ketersediaan rumah sakit dengan layanan dan fasilitas yang memadai di setiap daerah menjadi indikator utama dalam mengukur kualitas pelayanan kesehatan suatu negara. Di Indonesia, berdasarkan data Kementerian Kesehatan, terdapat lebih dari 3.000 rumah sakit dengan beragam jenis seperti Rumah Sakit Umum, Khusus, TNI/Polri, dan lainnya yang tersebar di seluruh provinsi [[1](https://kemkes.go.id/app_asset/file_content_download/172231123666a86244b83fd8.51637104.pdf)].
Namun, penyebaran serta tipe rumah sakit tersebut tidak selalu berbanding lurus dengan kebutuhan dan jumlah penduduk di daerah tertentu. Analisis terhadap jenis rumah sakit berdasarkan sumber daya yang dimiliki seperti jumlah dokter, perawat, bidan, tempat tidur, dan tenaga pendukung lainnya sangat penting untuk mengidentifikasi kesenjangan layanan kesehatan di wilayah Indonesia.

Dengan memanfaatkan pendekatan *machine learning*, kita dapat membangun model prediktif untuk mengklasifikasikan tipe rumah sakit berdasarkan data kuantitatif yang tersedia. Model ini berpotensi membantu lembaga terkait dalam pengambilan keputusan, baik untuk kebijakan distribusi tenaga kesehatan maupun peningkatan layanan rumah sakit di wilayah yang membutuhkan.

Beberapa studi internasional telah menunjukkan keberhasilan pendekatan *data mining* dan *machine learning* dalam sektor kesehatan. Misalnya, penelitian oleh Ibrahim et al. (2022) menunjukkan bahwa klasifikasi rumah sakit berdasarkan profil operasional dapat meningkatkan efisiensi perencanaan sistem kesehatan nasional [[2](https://link.springer.com/article/10.1007/s10916-022-01826-4)]. Dengan demikian, proyek ini diharapkan dapat memberikan kontribusi nyata terhadap pemanfaatan data kesehatan di Indonesia secara lebih cerdas dan terarah.

## Business Understanding

### Problem Statements
- Bagaimana memanfaatkan *machine learning* untuk memprediksi **kelas rumah sakit (A, B, C, D)** berdasarkan data sumber daya dan fasilitas rumah sakit di Indonesia?
- Algoritma *machine learning* apa yang paling efektif dalam **mengklasifikasikan kelas rumah sakit** dari data yang tersedia?
- Apakah terdapat **ketidakseimbangan kelas** dalam distribusi rumah sakit berdasarkan kelasnya, dan bagaimana performa model dalam menghadapinya?
- Fitur-fitur apa saja yang paling berpengaruh terhadap **penentuan kelas rumah sakit**, menurut hasil analisis dan pemodelan?

### Goals
- Membangun model *machine learning* untuk **memprediksi kelas rumah sakit (A, B, C, atau D)** berdasarkan karakteristik kuantitatif seperti jumlah dokter, perawat, bidan, tempat tidur, dan tenaga kesehatan lainnya.
- Membandingkan performa beberapa algoritma klasifikasi untuk menentukan model terbaik dalam mengklasifikasikan kelas rumah sakit.
- Mengidentifikasi fitur-fitur penting yang berkontribusi terhadap klasifikasi kelas rumah sakit.
- Menangani potensi **ketidakseimbangan jumlah data tiap kelas** dengan teknik *oversampling* (misalnya **SMOTE**) guna meningkatkan performa model.

### Solution statements
- Menerapkan algoritma **Decision Tree, Random Forest,** dan **K-Nearest Neighbors (KNN)** sebagai *baseline* model klasifikasi kelas rumah sakit.
- Menggunakan metrik evaluasi seperti *accuracy*, *precision*, *recall*, dan *F1-score* untuk menilai performa model.
- Menerapkan teknik ***oversampling*** **seperti SMOTE** apabila ditemukan distribusi kelas yang tidak seimbang.
- Melakukan **visualisasi** dan ***exploratory data analysis (EDA)*** untuk memahami pola data, distribusi fitur, dan korelasinya dengan kelas rumah sakit.


## Data Understanding

Dataset yang digunakan berasal dari [Kaggle - Klasifikasi Rumah Sakit](https://www.kaggle.com/datasets/muhammadhabibna/hospital-data-in-indonesia/data), yang berisi data rumah sakit di Indonesia.

**Jumlah data:** 3155baris  
**Jumlah fitur:** 12 kolom  
**Struktur dataset** :
![Struktur Dataset](https://drive.google.com/uc?export=view&id=1Sv3cu3Ca48gryj7JYV1-FxGXMiEwizey)


### Deskripsi Variable:

| Nama Variabel         | Tipe Data | Deskripsi                                                                 |
|-----------------------|-----------|---------------------------------------------------------------------------|
| `id`                  | int64     | ID unik untuk setiap rumah sakit.                                         |
| `nama`                | object    | Nama rumah sakit.                                                         |
| `propinsi`            | object    | Provinsi tempat rumah sakit berada.                                       |
| `kab`                 | object    | Kabupaten/Kota tempat rumah sakit berada.                                 |
| `alamat`              | object    | Alamat lengkap rumah sakit.                                               |
| `jenis`               | object    | Jenis rumah sakit (misalnya: Umum, Khusus, TNI/Polri, dll).              |
| `kelas`               | object    | Kelas rumah sakit (A, B, C, D atau tidak diklasifikasikan).               |
| `status_blu`          | object    | Status Badan Layanan Umum (BLU) dari rumah sakit.                         |
| `kepemilikan`         | object    | Kepemilikan rumah sakit (Pemerintah, Swasta, TNI/Polri, dll).            |
| `total_tempat_tidur`  | int64     | Jumlah total tempat tidur yang tersedia di rumah sakit.                   |
| `total_layanan`       | int64     | Jumlah total layanan kesehatan yang disediakan oleh rumah sakit.          |
| `total_tenaga_kerja`  | int64     | Jumlah total tenaga kerja di rumah sakit, termasuk dokter dan perawat.   |



### EDA Univariate:
Distribusi histogram fitur numerik
![Gambar Histogram](https://drive.google.com/uc?export=view&id=1gBtRxtd3ToVJ4PgbSKux_Zs7Oo0JDa8_)


- Distribusi total_tempat_tidur: Distribusi ini bersifat multimodal dengan beberapa puncak yang jelas terlihat, dan miring ke kanan (positively skewed). Ini menunjukkan bahwa mayoritas rumah sakit memiliki jumlah tempat tidur yang rendah hingga sedang, sementara hanya sebagian kecil yang memiliki jumlah tempat tidur sangat tinggi (hingga 300), yang menarik ekor distribusi ke kanan. Kondisi ini mungkin mencerminkan perbedaan kapasitas antara rumah sakit kecil dan rumah sakit besar atau pusat rujukan nasional.

- Distribusi total_layanan: Distribusi total layanan tampak miring ke kanan (positively skewed). Sebagian besar rumah sakit menawarkan layanan dalam jumlah sedang (sekitar 20–40), sementara terdapat beberapa rumah sakit dengan jumlah layanan yang jauh lebih banyak, yang menyebabkan distribusi memiliki ekor panjang di sebelah kanan. Hal ini bisa jadi mencerminkan rumah sakit besar atau pusat kesehatan khusus dengan banyak layanan spesialis.

- Distribusi total_tenaga_kerja: Distribusi ini sangat miring ke kanan (highly positively skewed). Mayoritas rumah sakit memiliki jumlah tenaga kerja yang relatif kecil hingga sedang, namun ada segelintir rumah sakit dengan jumlah tenaga kerja sangat besar (hingga lebih dari 700 orang). Nilai ekstrem ini mencerminkan keberadaan rumah sakit besar dengan skala operasional yang luas dan tenaga medis yang banyak.

Distribusi kolom kategorik
![Gambar Histogram](https://drive.google.com/uc?export=view&id=10tTcTYF_VC3RrSq_p8M4LB3H_13ztH7u)


Insight Distribusi Rumah Sakit per Provinsi di Indonesia:
1. Pola Konsentrasi Rumah Sakit
- Provinsi Jawa Timur, Jawa Barat, dan Jawa Tengah mendominasi jumlah rumah sakit di Indonesia. Ketiga provinsi ini memiliki jumlah rumah sakit lebih dari 300, menunjukkan konsentrasi layanan kesehatan yang tinggi di Pulau Jawa.
- Sumatera Utara dan DKI Jakarta juga memiliki jumlah rumah sakit yang tinggi, mengindikasikan tingginya kebutuhan layanan kesehatan di daerah dengan kepadatan penduduk yang besar.

2. Ketimpangan Distribusi Antar Wilayah
- Provinsi-provinsi di bagian timur Indonesia seperti Papua, Papua Barat, Papua Pegunungan, dan Papua Selatan memiliki jumlah rumah sakit yang sangat sedikit (kurang dari 20). Hal ini menunjukkan adanya kesenjangan akses layanan kesehatan antar wilayah barat dan timur Indonesia.
- Provinsi seperti Nusa Tenggara Timur, Maluku, dan Sulawesi Barat juga memiliki jumlah rumah sakit yang relatif rendah dibandingkan dengan provinsi di Jawa dan Sumatera.

3. Provinsi dengan Jumlah Rumah Sakit Terendah
- Provinsi seperti Papua Pegunungan, Papua Selatan, dan Papua Barat Daya tercatat sebagai wilayah dengan jumlah rumah sakit paling sedikit di Indonesia, menandakan potensi masalah dalam aksesibilitas layanan kesehatan dasar di wilayah ini.

4. Indikasi Perluasan Fasilitas
- Wilayah-wilayah dengan jumlah rumah sakit terbatas kemungkinan memerlukan intervensi kebijakan berupa pembangunan rumah sakit baru atau distribusi ulang tenaga kesehatan.
- Pemerataan fasilitas kesehatan ini sangat krusial untuk memastikan keadilan sosial dan pemerataan pelayanan publik di seluruh wilayah Indonesia.

### EDA Multivariate:
![Klasifikasi Kelas Rumah Sakit](https://drive.google.com/uc?export=view&id=17vfdrxijC5pM7fu8Zu1qy7y-kq8s1a3B)

**Insight Distribusi Kelas Rumah Sakit per Provinsi**
1. Dominasi Kelas C secara Nasional

* Kelas C secara konsisten mendominasi di hampir semua provinsi, terutama di
Jawa Timur (sekitar 240 RS kelas C) &
Jawa Barat, Jawa Tengah, dan DKI Jakarta
* Ini mengindikasikan bahwa rumah sakit dengan fasilitas tingkat menengah adalah yang paling umum secara nasional.

2. Distribusi Kelas A Sangat Terbatas
* Hanya beberapa provinsi yang memiliki rumah sakit kelas A dalam jumlah signifikan:
DKI Jakarta, Jawa Barat, Jawa Tengah, dan Jawa Timur
* Banyak provinsi sama sekali tidak memiliki rumah sakit kelas A, terutama di wilayah timur seperti: Papua, Papua Selatan, Papua Tengah, Papua Pegunungan, dan Maluku Utara
* Insight penting: Fitur provinsi sangat penting dalam mendeteksi kemungkinan kelas A — provinsi di luar Pulau Jawa sangat kecil peluangnya.

3. Kelas D Menjadi Penopang di Daerah Kurang Berkembang
* Kelas D cukup dominan di wilayah seperti:
Papua, Nusa Tenggara Timur, Maluku, Kalimantan Barat, dan Sulawesi Tenggara
* Biasanya mewakili rumah sakit kecil atau daerah terpencil.
* Dalam klasifikasi, provinsi-provinsi ini dapat menjadi indikator kuat untuk prediksi kelas D.

4. Kelas B Cenderung Menengah ke Atas dan Terpusat
* Kelas B ditemukan cukup banyak di: DKI Jakarta, Jawa Timur, Jawa Tengah, dan Jawa Barat.
* Juga muncul dalam jumlah kecil di provinsi urban lainnya seperti:
Sumatera Utara, Riau, Sulawesi Selatan.
* Insight penting: Provinsi besar di luar Jawa juga tidak bisa diabaikan sepenuhnya untuk kelas B.

**Visualisasi Pairplot**
![Gambar Pairplot](https://drive.google.com/uc?export=view&id=1wVITj35PGj49iKnoDbZz8LhD3xEiPbvn)

Gambar di atas menunjukkan distribusi dan hubungan antara tiga variabel penting:
* total_tempat_tidur (jumlah tempat tidur)
* total_layanan (jumlah layanan yang disediakan)
* total_tenaga_kerja (jumlah tenaga kerja)

Insight:
* Terlihat adanya korelasi positif antara jumlah tempat tidur, jumlah layanan, dan tenaga kerja.
* Distribusi total_tempat_tidur menunjukkan pola bimodal, menandakan dua kelompok utama rumah sakit berdasarkan kapasitas.
* Korelasi antara total_tenaga_kerja dan total_tempat_tidur tampak cukup kuat, menunjukkan bahwa rumah sakit yang lebih besar cenderung mempekerjakan lebih banyak tenaga kerja.


**Visualisasi heatmap korelasi**
![Gambar Heatmap Korelasi](https://drive.google.com/uc?export=view&id=1aJK_I_E5mcZswURwhn9-sQVEkg5iJ7Tt)

Heatmap ini menunjukkan hubungan linear antar fitur numerik dalam dataset, termasuk terhadap target kelas. Nilai korelasi berkisar antara -1 sampai 1:
* 1 = hubungan positif sempurna
* -1 = hubungan negatif sempurna
* 0 = tidak ada hubungan linear

| Pasangan Fitur                                  | Korelasi  | Interpretasi                                                                                                                                                                                                                       |
| ----------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **total\_tempat\_tidur & total\_tenaga\_kerja** | **+0.67** | Korelasi **cukup kuat positif**. Rumah sakit dengan lebih banyak tempat tidur cenderung memiliki lebih banyak tenaga kerja. Hal ini **masuk akal operasional**, karena kapasitas lebih besar butuh SDM lebih banyak.               |
| **total\_tempat\_tidur & total\_layanan**       | **+0.56** | Korelasi **sedang positif**. Artinya makin banyak tempat tidur, layanan medis yang tersedia juga cenderung meningkat. Menunjukkan **ekspansi fasilitas** beriringan dengan peningkatan layanan.                                    |
| **total\_tenaga\_kerja & total\_layanan**       | **+0.50** | Korelasi **cukup positif**, meskipun lebih rendah dari yang lain. Bisa diartikan bahwa meskipun jumlah layanan bertambah, **tidak selalu secara proporsional menambah tenaga kerja** (mungkin karena efisiensi atau spesialisasi). |


## Data Preparation
- Handling missing values: tidak ditemukan adanya missing values
- Handling duplicate values: tidak ditemukan adanya duplicate values
- Handling Outlier: dilakukan menggunakan teknik IQR Method untuk mengurangi pengaruh data ekstrem yang bisa memengaruhi akurasi model, terutama pada algoritma yang sensitif terhadap nilai outlier seperti Decision Tree dan Random Forest.
- Label Encode untuk kolom kelas: karena kelas merupakan data kategorikal, sehingga perlu diubah menjadi representasi numerik biner agar dapat diproses oleh model.
- Menghapus kolom provinsi, kab, jenis, status_blu, kepemilikan: Memfokuskan fitur numerik yang akan digunakan untuk melatih model
- Memisahkan kolom x dan y: memisahkan fitur (X) dan target (y) guna menyiapkan data sebelum pelatihan model klasifikasi dilakukan
- Split data train dan test dengan rasio 80:20: membagi data menjadi data latih dan data uji dengan rasio 80:20 untuk pelatihan dan evaluasi model.
- Data scaling: menggunakan StandardScaler untuk menstandarkan skala fitur numerik sehingga memiliki distribusi mean 0 dan standard deviasi 1. Ini penting karena model tertentu sensitif terhadap perbedaan skala fitur.
- Oversampling SMOTE pada data latih (train): untuk menangani ketidakseimbangan kelas (class imbalance), sehingga jumlah data pada kelas minoritas diseimbangkan dengan kelas mayoritas. Hal ini dapat meningkatkan sensitivitas model dalam mendeteksi kelas minoritas.

**Alasan**
- Handling missing values diperiksa untuk memastikan tidak ada data kosong yang dapat mengganggu hasil model atau analisis.
- Handling duplicate values diperiksa agar tidak ada duplikasi data yang bisa membuat bobot informasi menjadi tidak proporsional.
- Handling Outlier (IQR Method) dilakukan untuk mengurangi pengaruh data ekstrem yang bisa mendistorsi parameter model, terutama untuk algoritma berbasis pohon (Decision Tree, Random Forest) yang cukup sensitif terhadap outlier. Dengan membersihkan outlier, model bisa lebih stabil dan akurat.
- Label Encode untuk kolom kategorikal (kelas) Karena algoritma machine learning tidak dapat memproses data kategorikal dalam format string, maka perlu diubah menjadi format numerik biner agar bisa diproses dengan benar.
- Menghapus beberapa kolom agar memudahkan dalam pengodingan untuk kolom y
- Memisahkan kolom x(fitur) dan kolom y(label/target) dimana y hanya berisi kolom kelas yang terdiri dari kelas A, kelas B, kelas C, kelas D sedangkan x adalah fitur selain kelas
- Split data traint dan test tentu dilakukan agar bisa melakukan proses pelatihan dan evaluasi model
- Data Scaling (StandardScaler) penting untuk model-model seperti K-Nearest Neighbour dan algoritma berbasis jarak lainnya, di mana perbedaan skala antar fitur bisa menyebabkan fitur dengan rentang nilai lebih besar mendominasi hasil model.
- Oversampling dengan SMOTE pada data latih dilakukan untuk menangani class imbalance yang bisa membuat model cenderung bias ke kelas mayoritas. Dengan menyeimbangkan jumlah data minoritas, model bisa lebih sensitif dalam mendeteksi kategori kelas A. SMOTE hanya diterapkan ke data training, agar evaluasi di data testing tetap mencerminkan kondisi nyata dari distribusi data asli.


## Model Development
Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:

### **K-Nearest Neighbour**  
Model pertama yang saya gunakan adalah algoritma K-Nearest Neighbors (KNN), yaitu metode klasifikasi yang memprediksi label suatu data baru berdasarkan label dari K data terdekatnya dalam ruang fitur. Kedekatan antar data biasanya diukur menggunakan metrik jarak, seperti jarak Euclidean.

Cara kerja KNN dapat dijelaskan sebagai berikut:

1. Menentukan Nilai K: Tentukan jumlah tetangga terdekat yang akan digunakan dalam proses klasifikasi. Pada kasus ini, saya menggunakan nilai K = 5.

2. Menghitung Jarak: Hitung jarak antara data baru dengan seluruh data dalam dataset pelatihan menggunakan metrik jarak tertentu.

3. Menentukan K Tetangga Terdekat: Identifikasi K data pelatihan yang memiliki jarak paling dekat dengan data baru tersebut.

4. Melakukan Klasifikasi: Dari K tetangga terdekat, hitung frekuensi kemunculan setiap label kelas. Kelas yang paling sering muncul di antara tetangga tersebut akan menjadi prediksi label untuk data baru.

Parameter yang digunakan:
- n_neighbors jumlah tetangga terdekat yang digunakan untuk menentukan kelas prediksi. Dalam proyek ini ditentukan sebanyak 5 tetangga.

Kelebihan:
- Mudah dipahami dan diimplementasikan.
- Cocok untuk dataset skala kecil hingga menengah.
- Tidak memerlukan proses training yang kompleks.

Kekurangan:
- Proses prediksi lambat pada dataset besar karena harus menghitung jarak ke seluruh data latih.
- Sensitif terhadap skala fitur (perlu scaling).
- Rentan terhadap outlier dan noise.


### **Decision Tree Classifier**
Selanjutnya, saya menggunakan algoritma Decision Tree, yaitu metode pembelajaran terawasi (supervised learning) yang dapat digunakan untuk menyelesaikan masalah klasifikasi maupun regresi. Algoritma ini membangun struktur menyerupai pohon, di mana:

* Setiap node internal mewakili suatu fitur atau atribut dari data,

* Setiap cabang mewakili kondisi atau aturan pengambilan keputusan berdasarkan nilai fitur tersebut,

* Dan setiap node daun (leaf) menunjukkan hasil akhir berupa label kelas atau nilai prediksi.

Tujuan utama dari Decision Tree adalah membentuk rangkaian aturan keputusan secara bertahap dalam bentuk pernyataan "jika-maka" (if-then), yang dapat digunakan untuk mengklasifikasikan data baru atau memprediksi nilainya berdasarkan atribut yang dimilikinya.

Parameter yang digunakan:
- random_state digunakan untuk memastikan bahwa proses acak (seperti pemilihan subset data atau fitur saat membangun node) menghasilkan output yang konsisten setiap kali kode dijalankan; dalam proyek ini nilainya ditetapkan ke 42
- max_depth membatasi kedalaman maksimum pohon dalam model untuk menghindari overfitting, dan pada proyek ini diatur hingga maksimum 3.

Kelebihan:
- Mudah dipahami dan divisualisasikan.
- Dapat menangani data numerik maupun kategorikal tanpa perlu normalisasi.
- Dapat mengukur pentingnya tiap fitur (feature importance).

Kekurangan:
- Rentan terhadap overfitting jika tidak dibatasi kedalamannya.
- Performa bisa kurang stabil jika dataset kecil atau banyak noise.
- Cenderung bias ke fitur dengan banyak nilai unik.

### **Random Forest Classifier**  
Terakhir, saya menggunakan algoritma Random Forest, sebuah metode supervised learning yang termasuk dalam kategori ensemble learning. Berbeda dengan pendekatan yang hanya mengandalkan satu model, Random Forest membangun sejumlah pohon keputusan (Decision Tree) secara independen. Setelah semua pohon terbentuk, hasil prediksi dari masing-masing pohon kemudian digabungkan untuk menghasilkan prediksi akhir.

Konsep dasar dari Random Forest adalah bahwa gabungan dari banyak model sederhana (dalam hal ini, pohon keputusan yang relatif lemah) dapat membentuk sebuah model yang lebih kuat, stabil, dan akurat. Pendekatan ini membantu mengurangi risiko overfitting yang umum terjadi pada model Decision Tree tunggal.

Parameter yang digunakan:
- random_state digunakan untuk memastikan bahwa proses acak (seperti pemilihan subset data atau fitur saat membangun node) menghasilkan output yang konsisten setiap kali kode dijalankan; dalam proyek ini nilainya ditetapkan ke 42
- max_depth membatasi kedalaman maksimum pohon dalam model untuk menghindari overfitting, dan pada proyek ini diatur hingga maksimum 3.


Kelebihan:
- Lebih stabil dan akurat dibanding single Decision Tree.
- Mampu mengatasi overfitting karena melakukan averaging dari banyak model.
- Bisa mengukur feature importance.
- Tahan terhadap outlier dan noise.

Kekurangan:
- Proses training dan prediksi lebih lambat karena banyaknya pohon yang dibuat.
- Model cenderung sulit diinterpretasi karena kompleksitas ensemble.
- Membutuhkan lebih banyak resource memori dan komputasi.


## Evaluation
Dalam tahap evaluasi, metrik yang digunakan adalah

- **Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision**
  
$$\text{Precision} = \frac{TP}{TP + FP}$$

- **Recall**

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **F1-Score**

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Penjelasan
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).


**Hasil akhir:**

Berikut adalah hasil evaluasi dari tiga model machine learning yang digunakan: KNN, Decision Tree, dan Random Forest. Metrik yang digunakan untuk penilaian meliputi accuracy, precision, recall, dan f1-score.

| Metric    | KNN      | Decision Tree | Random Forest |
|-----------|----------|---------------|---------------|
| Accuracy  | 0.698885 | 0.620818      | 0.596654      |
| Precision | 0.781984 | 0.839560      | 0.765386      |
| Recall    | 0.698885 | 0.620818      | 0.596654      |
| F1 Score  | 0.727167 | 0.695723      | 0.611702      |

Insight Evaluasi Model

1. Model dengan Akurasi Tertinggi adalah K-Nearest Neighbors (KNN) dengan akurasi 69,89%, diikuti oleh Decision Tree dan Random Forest. Ini menunjukkan bahwa KNN lebih konsisten dalam memprediksi kelas rumah sakit secara keseluruhan.

2. Precision Tertinggi justru dimiliki oleh Decision Tree (83,96%), menunjukkan bahwa model ini lebih andal dalam menghindari prediksi yang salah untuk kelas minoritas — meskipun akurasinya lebih rendah. Hal ini bisa menunjukkan ketimpangan dalam distribusi kelas, di mana model Decision Tree lebih konservatif tetapi “tepat sasaran” ketika memberikan prediksi positif.

3. Recall dan F1-Score mengikuti tren yang sama dengan akurasi:

* Recall dan F1-Score tertinggi dimiliki oleh KNN, memperkuat bahwa model ini paling seimbang antara ketepatan dan kelengkapan dalam prediksi.

* Random Forest menunjukkan performa terendah di seluruh metrik, mengindikasikan bahwa konfigurasi atau tuning-nya mungkin belum optimal untuk dataset ini.

4. Kesimpulan Model Terbaik:

* KNN adalah model terbaik secara keseluruhan, karena memberikan keseimbangan terbaik antara accuracy, recall, dan f1-score.

* Decision Tree bisa dipertimbangkan apabila fokus analisis lebih berat pada precision (misalnya, jika salah klasifikasi lebih berdampak daripada tidak mengklasifikasikan).

* Random Forest memerlukan tuning lebih lanjut atau bisa digantikan, karena saat ini memiliki performa paling rendah.

**Kesimpulan**
1. Pemanfaatan Machine Learning untuk Prediksi Kelas Rumah Sakit
  Model machine learning dapat dimanfaatkan secara efektif untuk Klasifikasi Kelas di Rumah Sakit berdasarkan variabel seperti total_tempat_tidur, total_layanan, total_tenaga_kerja . Hasil model menunjukkan adanya pola yang dapat digunakan untuk memisahkan kategori Kelas A,B,C,D.

2. Algoritma Machine Learning yang Paling Efektif
  Dari tiga algoritma baseline yang diuji, yaitu K-Nearest Neighbour, Decision Tree, dan Random Forest, model  K-Nearest Neighbour menunjukkan performa paling stabil dan unggul dalam hal accuracy, recall, dan F1-score. Model ini mampu menangani ketidakseimbangan data lebih baik, apalagi setelah diterapkan SMOTE (Synthetic Minority Oversampling Technique).

3. Performa Model setelah Oversampling SMOTE
  Penerapan SMOTE berhasil meningkatkan performa model secara signifikan, khususnya dalam hal recall untuk kategori Kelas. Hal ini menunjukkan bahwa teknik oversampling efektif mengatasi ketidakseimbangan kelas dalam dataset.

4. Fitur-Fitur yang Paling Berpengaruh Hasil Exploratory Data Analysis (EDA) dan analisis korelasi menunjukkan bahwa fitur total_tenaga_kerja memiliki korelasi paling kuat terhadap kelas. Disusul oleh total_tempat_tidur dan total_layanan yang memiliki hubungan signifikan terhadap klasifikasi kelas pada Rumah Sakit.

**Referensi**
1. Obeid, N., Aljarah, I., & Faris, H. (2021). *Poverty Classification Using Machine Learning: The Case of Jordan*. Diakses dari [https://www.researchgate.net/publication/348898452\_Poverty\_Classification\_Using\_Machine\_Learning\_The\_Case\_of\_Jordan](https://www.researchgate.net/publication/348898452_Poverty_Classification_Using_Machine_Learning_The_Case_of_Jordan)

2. Subramanian, D. (2019, November 3). *A Simple Introduction to K-Nearest Neighbors Algorithm*. Towards Data Science. Diakses dari [https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e](https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e)

3. IBM. (n.d.). *What are Decision Trees?*. Diakses dari [https://www.ibm.com/think/topics/decision-trees](https://www.ibm.com/think/topics/decision-trees)

4. Wood, T. (n.d.). *What is a Random Forest?*. DeepAI. Diakses dari [https://deepai.org/machine-learning-glossary-and-terms/random-forest](https://deepai.org/machine-learning-glossary-and-terms/random-forest)

---
**Catatan:**  
EDA visualisasi, Data Cleaning & Preprocessing ,Confusion Matrix, dan proses training dapat dilihat langsung di notebook terlampir.
