
##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem : Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

# Telco müşteri churn verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan
# hayali bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu içermektedir.

# 21 Değişken 7043 Gözlem

# CustomerId : Müşteri İd’si
# Gender : Cinsiyet
# SeniorCitizen : Müşterinin yaşlı olup olmadığı (1, 0)
# Partner : Müşterinin bir ortağı olup olmadığı (Evet, Hayır) ? Evli olup olmama
# Dependents : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır) (Çocuk, anne, baba, büyükanne)
# tenure : Müşterinin şirkette kaldığı ay sayısı
# PhoneService : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV : Müşterinin, bir üçüncü taraf sağlayıcıdan televizyon programları yayınlamak için İnternet hizmetini kullanıp kullanmadığını gösterir
# StreamingMovies :  Müşterinin bir üçüncü taraf sağlayıcıdan film akışı yapmak için İnternet hizmetini kullanıp kullanmadığını gösterir
# Contract : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges : Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges : Müşteriden tahsil edilen toplam tutar
# Churn : Müşterinin kullanıp kullanmadığı (Evet veya Hayır) - Geçen ay veya çeyreklik içerisinde ayrılan müşteriler


# Her satır benzersiz bir müşteriyi temsil etmekte.
# Değişkenler müşteri hizmetleri, hesap ve demografik veriler hakkında bilgiler içerir.
# Müşterilerin kaydolduğu hizmetler - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Müşteri hesap bilgileri – ne kadar süredir müşteri oldukları, sözleşme, ödeme yöntemi, kağıtsız faturalandırma, aylık ücretler ve toplam ücretler
# Müşteriler hakkında demografik bilgiler - cinsiyet, yaş aralığı ve ortakları ve bakmakla yükümlü oldukları kişiler olup olmadığı


# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
           # Adım 3:  Numerik ve kategorik değişkenlerin analizini yapınız.
           # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
           # Adım 5: Aykırı gözlem analizi yapınız.
           # Adım 6: Eksik gözlem analizi yapınız.
           # Adım 7: Korelasyon analizi yapınız.

# GÖREV 2: FEATURE ENGINEERING
           # Adım 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız.
           # işlemleri uygulayabilirsiniz.



# Gerekli Kütüphane ve Fonksiyonlar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("3_Feature_Engineering&Data_Pre_Processing/7.week/Case_study/TelcoCustomerChurn-230423-212029.csv")
df.head()
df.shape
df.info()

# TotalCharges sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x : 1 if x == "Yes" else 0)
df.head()

##################################
# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car



##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)


# fonkisyon update
def cat_summary2(dataframe, cat_lst, plot=False):
    for col in cat_lst:
        print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col], data=dataframe)
            plt.show(block=True)

cat_summary2(df, cat_cols)



# Veri setimizdeki müşterilerin yaklaşık yarısı erkek, diğer yarısı kadındır.
# Müşterilerin yaklaşık %50'sinin bir ortağı var (evli)
# Toplam müşterilerin yalnızca %30'unun bakmakla yükümlü olduğu kişiler var.
# Müşterilerin %90'u telefon hizmeti almaktadır.
# Telefon hizmeti alan %90'lık kesimin  yüzde 53'ü birden fazla hatta sahip değil
# Internet servis sağlayıcısı bulunmayan %21'lik bir kesim var
# Müşterilerin çoğu aydan aya sözleşme yapıyor. 1 yıllık ve 2 yıllık sözleşmelerde yakın sayıda  müşteri bulunmakta.
# Müşterilerin %60 i kağıtsız faturası bulunmakta
# Müşterilerin yaklaşık %26'sı geçen ay platformdan ayrılmış
# Veri setinin  %16'sı yaşlı  müşterilerden oluşmaktadır Dolayısıyla verilerdeki müşterilerin çoğu genç


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Tenure'e bakıldığında 1 aylık müşterilerin çok fazla olduğunu
# ardından da 70 aylık müşterilerin geldiğini görüyoruz.





##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

# Tenure ve Churn ilişkisine baktığımızda churn olmayan müşterilerin daha uzun süredir müşteri olduklarını görüyoruz
# monthlycharges ve Churn incelendiğinde churn olan müşterilerin ortalama aylık ödemeleri daha fazla



##################################
# KATEGORİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


# Kadın ve erkeklerde churn yüzdesi neredeyse eşit
# Partner ve dependents'i olan müşterilerin churn oranı daha düşük
# PhoneServise ve MultipleLines'da fark yok
# Fiber Optik İnternet Servislerinde kayıp oranı çok daha yüksek
# No OnlineSecurity , OnlineBackup ve TechSupport gibi hizmetleri olmayan müşterilerin churn oranı yüksek
# Bir veya iki yıllık sözleşmeli Müşterilere kıyasla, aylık aboneliği olan Müşterilerin daha büyük bir yüzdesi churn
# Kağıtsız faturalandırmaya sahip olanların churn oranı daha fazla
# ElectronicCheck PaymentMethod'a sahip müşteriler, diğer seçeneklere kıyasla platformdan daha fazla ayrılma eğiliminde
# Yaşlı müşterilerde churn yüzdesi daha yüksektir

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.corrwith(df["Churn"]).sort_values(ascending=False)

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

df["TotalCharges"].fillna(0, inplace=True)

df.isnull().sum()



##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


msno.bar(df)
plt.show(block=True)

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)

#Adım 2: Yeni değişkenler oluşturunuz.
check_df(df)
df["tenure"].hist()
plt.show(block=True)

df.head()
df.groupby("tenure")["TotalCharges"].mean().head()
#### 1.Degisken

mylabels = [0, 1, 2, 3, 4, 5]

bins = [-1, 12, 24, 36, 48, 60, 72]

df["NEW_TENURE_GROUP"] = pd.cut(df["tenure"], bins, labels=mylabels)


df["NEW_TENURE_GROUP"] = df["NEW_TENURE_GROUP"].astype("int64")
df["NEW_TENURE_GROUP"].dtype
df["NEW_TENURE_GROUP"].values
df["NEW_TENURE_GROUP"].isnull().sum()
df["tenure"].isnull().sum()

#### 2.Degisken
df["NEW_TENURE_GROUP_RATIO"] = df["NEW_TENURE_GROUP"] / df["MonthlyCharges"]

##### 3.Degisken
df["Contract"].values
##['Month-to-month', 'One year', 'Month-to-month', ...,
##   'Month-to-month', 'Month-to-month', 'Two year'

SENIOR_COSTUMER = pd.Series(["no_senior_monthly", "no_senior_yearly", "no_senior_two_year",
                                     "senior_monthly", "senior_yearly", "senior_two_year"], dtype = "category")

df["SENIOR_COSTUMER"] = SENIOR_COSTUMER

df.loc[(df["SeniorCitizen"] == 0) & (df["Contract"]) == "Month-to-month", "SENIOR_COSTUMER"] = SENIOR_COSTUMER[0]
df.loc[(df["SeniorCitizen"] == 0) & (df["Contract"]) == "One year", "SENIOR_COSTUMER"] = SENIOR_COSTUMER[1]
df.loc[(df["SeniorCitizen"] == 0) & (df["Contract"]) == "Two year", "SENIOR_COSTUMER"] = SENIOR_COSTUMER[2]
df.loc[(df["SeniorCitizen"] == 1) & (df["Contract"]) == "Month-to-month", "SENIOR_COSTUMER"] = SENIOR_COSTUMER[3]
df.loc[(df["SeniorCitizen"] == 1) & (df["Contract"]) == "One year", "SENIOR_COSTUMER"] = SENIOR_COSTUMER[4]
df.loc[(df["SeniorCitizen"] == 1) & (df["Contract"]) == "Two year", "SENIOR_COSTUMER"] = SENIOR_COSTUMER[5]

#df.drop("CONTRACT_CLASSIFICATION", axis=1, inplace=True)

type(df["SENIOR_COSTUMER"])
df["SENIOR_COSTUMER"].index
df["SENIOR_COSTUMER"].dtype
df["SENIOR_COSTUMER"].size
df["SENIOR_COSTUMER"].ndim
df["SENIOR_COSTUMER"].values
type(df["SENIOR_COSTUMER"].values)
df["SENIOR_COSTUMER"].head()
df["SENIOR_COSTUMER"].tail()

##### 4.Degisken
GENDER_TENURE = pd.Series(["female_1_year", "female_2_year", "female_3_year",
                        "female_4_year", "female_5_year", "female_6_year",
                        "male_1_year", "male_2_year", "male_3_year",
                        "male_4_year", "male_5_year", "male_6_year",
                          ], dtype = "category")

df["GENDER_TENURE"] = GENDER_TENURE

df.loc[(df["gender"] == "Female") & (df["NEW_TENURE_GROUP"] == 1), "GENDER_TENURE"] = GENDER_TENURE[0]
df.loc[(df["gender"] == "Female") & (df["NEW_TENURE_GROUP"] == 2), "GENDER_TENURE"] = GENDER_TENURE[1]
df.loc[(df["gender"] == "Female") & (df["NEW_TENURE_GROUP"] == 3), "GENDER_TENURE"] = GENDER_TENURE[2]
df.loc[(df["gender"] == "Female") & (df["NEW_TENURE_GROUP"] == 4), "GENDER_TENURE"] = GENDER_TENURE[3]
df.loc[(df["gender"] == "Female") & (df["NEW_TENURE_GROUP"] == 5), "GENDER_TENURE"] = GENDER_TENURE[4]
df.loc[(df["gender"] == "Female") & (df["NEW_TENURE_GROUP"] == 6), "GENDER_TENURE"] = GENDER_TENURE[5]

df.loc[(df["gender"] == "Male") & (df["NEW_TENURE_GROUP"] == 1), "GENDER_TENURE"] = GENDER_TENURE[6]
df.loc[(df["gender"] == "Male") & (df["NEW_TENURE_GROUP"] == 2), "GENDER_TENURE"] = GENDER_TENURE[7]
df.loc[(df["gender"] == "Male") & (df["NEW_TENURE_GROUP"] == 3), "GENDER_TENURE"] = GENDER_TENURE[8]
df.loc[(df["gender"] == "Male") & (df["NEW_TENURE_GROUP"] == 4), "GENDER_TENURE"] = GENDER_TENURE[9]
df.loc[(df["gender"] == "Male") & (df["NEW_TENURE_GROUP"] == 5), "GENDER_TENURE"] = GENDER_TENURE[10]
df.loc[(df["gender"] == "Male") & (df["NEW_TENURE_GROUP"] == 6), "GENDER_TENURE"] = GENDER_TENURE[11]


##### 5.Degisken
df.columns = [col.upper() for col in df.columns]

service_cols = ['MULTIPLELINES', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', "PHONESERVICE"]

for col in service_cols:
    df[col+"_NEW"] = np.where(df[col] == "Yes", 1, 0)

##### 6.Degisken

df["INTERNETSERVICE_NEW"] = np.where(df["INTERNETSERVICE"] == "No", 0, 1)

##### 7.Degisken

df["PHONE_NET"] = np.where((df["PHONESERVICE_NEW"] == 1) & (df["INTERNETSERVICE_NEW"] == 0), 0,
                           np.where((df["PHONESERVICE_NEW"] == 0) & (df["INTERNETSERVICE_NEW"] == 1), 1, 2))

##### 8.Degisken
df["NET_SERVICE_RATE"] = df["STREAMINGMOVIES_NEW"] + df["ONLINESECURITY_NEW"] + df["ONLINEBACKUP_NEW"] +\
                         df["DEVICEPROTECTION_NEW"] + df["TECHSUPPORT_NEW"] + df["STREAMINGTV_NEW"]

df.drop(columns = ['MULTIPLELINES', 'ONLINESECURITY', 'ONLINEBACKUP', 'DEVICEPROTECTION',
                   'TECHSUPPORT', 'STREAMINGTV', 'STREAMINGMOVIES', "PHONESERVICE"], inplace = True)



# -LabelEncoder,Binary-

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


bin_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype not in ["int64", "float64"]]

for col in bin_cols:
    label_encoder(df, col)

df.head()
#### 3.Degisken -OHE-

df.nunique()
def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]


one_hot_encoder(df, ohe_cols).head()

df.head()
df.shape

df["EXTRA_PEOPLE"] = df["PARTNER"] + df["DEPENDENTS"]

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


new_cat_col = [col for col in df.columns[1:] if col not in num_cols and df[col].dtype != "float64"]
rare_analyser(df, "CHURN", new_cat_col)


ohe_cols = [col for col in df.columns if 14 >= df[col].nunique() > 2 and df[col].dtype in ["object", "category"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


dff = one_hot_encoder(df, ohe_cols, drop_first = False)
dff.head()
df.shape
dff.shape
dff.columns = [col.upper() for col in dff.columns]
dff.drop(columns = ["CUSTOMERID"], inplace = True)
cat_cols, num_cols, cat_but_car = grab_col_names(dff)
rare_analyser(dff, "CHURN", cat_cols[1:])

ss = StandardScaler()
for col in num_cols:
    dff[col] = ss.fit_transform(df[[col]])
dff.head()


import os
from sklearn.ensemble import RandomForestClassifier

y = dff["CHURN"]
X = dff.drop(["CHURN"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 17)

rf_model = RandomForestClassifier(random_state = 46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value",
                                                                           ascending = False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)