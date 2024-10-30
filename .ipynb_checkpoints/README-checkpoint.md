# Seattle AirBnB Listing Dataset (Rakamin Academy Final Project)
Dalam final project ini, kelompok kami ingin membuat machine learning yang bertujuan untuk memberikan wawasan yang diperlukan dan dapat membantu AirBnB dalam meningkatkan jumlah customer di Kota Seattle sehingga posisi mereka di pasar Kota Seattle semakin kuat. 

## Kelompok 2 ResNet
- Project Manager : Mohammad Fauzan
- Data Analyst    : Indah Mutiah Utami. MZ
- Data Engineer   : Yusuf Nafi Farhan
- Data Scientist  : Julian

## Daftar Isi
- [Prerequisites](#prerequisites)
- [Penjelasan Sebelum EDA](#penjelasan-sebelum-eda)
- [EDA](#eda)


## Prerequisites
1. Download data [here](https://drive.google.com/drive/folders/1q0uoNhUzHYL3TmhfwtFL-Xnb26rwOzRF?usp=sharing)
2. Clone repositori ini:
   ```bash
   git clone https://github.com/Podjan/ResNet.git

## Penjelasan Sebelum EDA
### Dataset
Ada tiga dataset pada pekerjaan kali ini. 
1. **Calendar** yang berisi tentang data penghasilan dari AirBnB selama setahun.
2. **Listings** yang berisi tentang data detail lengkap mengenai setiap listings.
3. **Reviews** yang berisi tentang ulasan dari setiap listing.

### Beberapa hal yang perlu diperhatikan
1. Penentuan kolom dan dataset yang diambil dilakukan diawal untuk meminimalisir resiko dan membuang yang tidak perlu.
2. Poin pertama bukan berarti selanjutnya tidak ada pengambilan keputusan untuk membuang kolom. Jika ada kolom yang korelasinya mendekati 1 kemungkinan akan dibuang juga.
3. Penentuan kolom dan dataset diawal juga didasari oleh goal dan objective yang ingin dicapai.

### Goal dan Objectives
#### Goal
Goals yang ingin dicapai dalam studi case ini adalah meningkatkan jumlah customer AirBnB di tahun berikutnya dengan memberikan wawasan yang akurat kepada pemilik AirBNB mengenai jenis properti yang paling diminati oleh tamu dan tingkat kepuasan para tamu, sehingga mereka dapat mengoptimalkan strategi pemasaran, penetapan harga, dan pengelolaan inventaris di tahun berikutnya.
#### Objectives
1. Memprediksi jumlah pengunjung di tahun berikutnya berdasarkan kualitas dan harga.
2. Meningkatkan tingkat kepuasan customer terhadap kualitas hotel yang pengunjungnya sepi.
3. Memprediksi harga hotel di tahun depan untuk memberikan rekomendasi harga yang tepat di setiap hotel.

### Keputusan
1. Hanya melakukan merge dataset calendar dan listings. Reviews tidak diikutsertakan karena dataset tidak mempunyai id. Selain itu dataset listings juga punya kolom rating sehingga untuk kepuasan dari listing sudah terwakilkan.
2. Membuat kolom day_price_status, weekly_price_status, dan monthly_price_status. Hal ini untuk melihat apakah orang yang menyewa listing tersebut membayar dengan harga per hari, per minggu, atau per bulan. Hal ini juga memudahkan untuk melakukan group terhadap harga dari setiap listing dan juga menentukan alasan dari adanya outlier.
3. Kolom yang berhubungan dengan harga seperti security_deposit, cleaning_fee, extra_people di dataset listing dibuang karena dari dataset calendar tidak menampilkan rinciannya seperti berapa tambahan orang yang menginap. Kalaupun dibuat rumus untuk menebak kira-kira pakai harga dasar yang mana (harga dasar disini yaitu price, weekly_price, atau monthly_price) maka kecenderungannya bisa salah karena tidak ada detail rincian dari orang yang menginap di tanggal masing-masing. Dataset calendar hanya menampilkan harga dan harga dasar tersebut berdasarkan price, weekly_price, dan monthly_price di dataset listing.

### Kolom yang diambil
#### calendar
- listing_id
- date
- available
- price (di rename jadi payment)
#### listings
- listing_id
- name
- host_id
- host_response_time
- host_response_rate
- host_acceptance_rate
- host_is_superhost
- host_identity_verified
- zipcode
- latitude
- longitude
- is_location_exact
- property_type
- room_type
- accommodates
- bathrooms
- bedrooms
- beds
- bed_type
- price
- weekly_price
- monthly_price
- guests_included
- minimum_nights
- maximum_nights
- review_scores_rating
- instant_bookable
- cancellation_policy
- require_guest_profile_picture
- require_guest_phone_verification

### Fungsi untuk membuat day_price_status, weekly_price_status, dan monthly_price_status
```python
def update_status(df):
    """Fungsi untuk memperbarui kolom Day Price Status, Weekly Price Status, dan Monthly Price Status berdasarkan kolom Payment."""
    df['day_price_status'] = df.apply(lambda row: 'Yes' if row['payment'] == row['price'] else 'No', axis=1)
    df['weekly_price_status'] = df.apply(lambda row: 'Yes' if row['payment'] == row['weekly_price'] else 'No', axis=1)
    df['monthly_price_status'] = df.apply(lambda row: 'Yes' if row['payment'] == row['monthly_price'] else 'No', axis=1)
    return df
```
Fungsi diatas adalah fungsi untuk membuat tiga kolom status. **update_status** digunakan saat ada perubahan di kolom payment (payment adalah kolom price dari dataset listings yang sudah di rename) sehingga kita tinggal panggil kembali fungsi tersebut.
    
## EDA
### Descriptive Statistic
Pada proses descriptive static ini, langkah pertama yang dilakukan adalah mengubah tipe data yang kurang sesuai adapun features atau kolom-kolom yang diubah datanya pada proses ini adalah :

1. Mengubah data yang bertipe string ke float
   Kolom-kolom yang bertipe string diubah ke float sebab kolom-kolom tersebut sangat berpengaruh saat proses descriptive statistic. Berikut adalah kolom-kolomnya :
   
   | Features             | dataset    |
   |----------------------|------------|
   | payment              | calendar   |
   | price                | listing_id |
   | weekly_price         | listing_id |
   | monthly_price        | listing_id |
   | host_response_rate   | listing_id |
   | host_acceptance_rate | listing_id |
   
2. Mengubah data yang bertipe integer ke string
   Kolom-kolom yang bertipe integer diubah ke string sebab kolom-kolom tersebut sangat berpengaruh saat proses univariate ataupun multivariate. Berikut adalah kolom-kolomnya:
   
   | Features             | dataset    |
   |----------------------|------------|
   | host_id              | listing_id |
   | price                | listing_id |
   | weekly_price         | listing_id |


Selain mengubah tipe data, hal yang peling penting dalam pengolahan data adalah mengetahui nilai-nilai kosong sebagai bahan pertimbangan apakah data tersebut harus kita buang atau pertahankan. Adapun features yang memiliki data kosong / nan adalah sbb:

   | Features                 | tot. data nan    |
   |--------------------------|------------------|
   | payment                  | 459028           |
   | host_response_time       | 190895           |
   | host_response_rate       | 190895           |
   | host_acceptance_rate     | 282145           |
   | host_is_superhost        | 730              |
   | host_identity_verified   | 730              |
   | zipcode                  | 2555             |
   | property_type            | 363              |
   | bathrooms                | 5840             |
   | bedrooms                 | 2190             |
   | beds                     | 365              |
   | weekly_price             | 660285           |
   | monthly_price            | 839865           |
   | review_scores_rating     | 236155           |
   


Berikut ini adalah statistik deskriptif dari dataset yang digunakan:

### Statistik Deskriptif Data Numerik

| Column                | Count       | Mean           | Std            | Min   | 25%   | 50%   | 75%   | Max     |
|-----------------------|-------------|----------------|----------------|-------|-------|-------|-------|---------|
| payment               | 934542      | 137.94         | 105.06         | 10.0  | 75.0  | 109.0 | 160.0 | 1650.0  |
| host_response_rate    | 1202675     | 9.85           | 11.86          | 1.0   | 7.0   | 10.0  | 10.0  | 100.0   |
| host_acceptance_rate  | 1111425     | 9.99           | 18.11          | 1.0   | 7.0   | 10.0  | 10.0  | 100.0   |
| accommodates          | 1393570     | 3.34           | 1.97           | 1.0   | 2.0   | 3.0   | 4.0   | 16.0    |
| bathrooms             | 1387730     | 1.25           | 0.59           | 0.0   | 1.0   | 1.0   | 2.0   | 8.0     |
| bedrooms              | 1391380     | 1.30           | 0.88           | 0.0   | 1.0   | 1.0   | 2.0   | 7.0     |
| beds                  | 1393205     | 1.73           | 1.13           | 1.0   | 1.0   | 1.0   | 2.0   | 15.0    |
| price                 | 1393570     | 127.97         | 90.28          | 22.0  | 100.0 | 150.0 | 75.0  | 6300.0  |
| weekly_price          | 733285      | 788.48         | 532.22         | 100.0 | 455.0 | 650.0 | 950.0 | 6300.0  |
| monthly_price         | 553705      | 2613.33        | 1721.70        | 500.0 | 1512.0| 2200.0| 3150.0| 19500.0 |
| guests_included       | 1393570     | 1.67           | 1.31           | 1.0   | 1.0   | 1.0   | 2.0   | 15.0    |
| minimum_nights        | 1393570     | 2.36           | 16.03          | 1.0   | 1.0   | 1.0   | 2.0   | 1000.0  |
| maximum_nights        | 1393570     | 780.45         | 1683.36        | 1.0   | 6.0   | 90.0  | 1125.0| 100000.0|
| review_scores_rating  | 1157415     | 94.53          | 6.60           | 2.0   | 93.0  | 96.0  | 99.0  | 100.0   |

Dari hasil descriptive statistik terlihat tidak ada nilai summary yang aneh, semua data hasilnya normal, tidak ada nilai minus atau pun data-data yang bermasalah


### Univariate Analysis

### Multivariate Analysis