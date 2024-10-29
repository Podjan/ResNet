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
- city
- state
- zipcode
- market
- smart_location
- country_code
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
Pada proses descriptive static ini, langkah pertama yang dilakukan adalah mengubah tipe data yang kurang sesuai agar tidak salah dalam melakukan pengolahan data. Beberapa data yang perlu disesuikan tipe datanya adalah payment, price, weekly_price, dan monthly_price diubah ke dalam tipe data float, latitude dan longitude diubah dalam bentuk string. Latitude dan logitude diubah dalam bentuk string sebab dalam proses descriptive statistic data tersebut kurang tepat jika diolah karena merepresentasikan letak sebuah penginapan. 


Berikut ini adalah statistik deskriptif dari dataset yang digunakan:

| Feature               | Count      | Mean      | Std Dev   | Min      | 25%      | 50%      | 75%      | Max      |
|-----------------------|------------|-----------|-----------|----------|----------|----------|----------|----------|
| listing_id            | 1,393,570  | 5,550,111 | 2,962,274 | 3,335    | 3,258,213| 6,118,244| 8,035,212| 10,340,167|
| payment               | 934,542    | 137.94    | 105.06    | 10       | 75       | 109      | 160      | 1,650     |
| host_id               | 1,393,570  | 15,785,566| 14,581,910| 4,193    | 3,271,389| 1,055,814| 2,590,413| 5,320,861 |
| accommodates          | 1,393,570  | 3.34      | 1.97      | 1        | 2        | 3        | 4        | 16        |
| bathrooms             | 1,387,730  | 1.25      | 0.59      | 0        | 1        | 1        | 2        | 8         |
| bedrooms              | 1,391,380  | 1.30      | 0.88      | 0        | 1        | 1        | 2        | 7         |
| beds                  | 1,393,205  | 1.75      | 1.19      | 1        | 1        | 1        | 2        | 15        |
| price                 | 1,393,570  | 127.97    | 90.28     | 20       | 75       | 100      | 150      | 1,000     |
| weekly_price          | 733,285    | 788.48    | 532.22    | 100      | 455      | 650      | 950      | 6,300     |
| monthly_price         | 553,705    | 2,613.34  | 1,721.70  | 500      | 1,512    | 2,200    | 3,150    | 19,500    |
| guests_included       | 1,393,570  | 1.67      | 1.31      | 1        | 1        | 1        | 2        | 15        |
| minimum_nights        | 1,393,570  | 2.36      | 16.30     | 1        | 1        | 1        | 1        | 1,000     |
| maximum_nights        | 1,393,570  | 78.04     | 168.34    | 1        | 6        | 30       | 112      | 100,000   |
| review_scores_rating  | 1,157,415  | 94.54     | 6.60      | 2        | 93       | 96       | 99       | 100       |


### Univariate Analysis

### Multivariate Analysis