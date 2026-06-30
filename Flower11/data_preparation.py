import pandas as pd

df = pd.read_csv('data/temp_database1.csv', sep=';')

#Data preprocessing

#Encoding of the target variable (if a patient has cancer or not: 1 if YES; O if NO)
df['case_csPCa'] = df['case_csPCa'].map({'YES': 1, 'NO': 0})

#Drop columns that are not important (dates, type of analysis...)
cols_to_keep = ['patient_age', 'psa', 'psad', 'prostate_volume','case_csPCa']
df = df[cols_to_keep]

#Adjusting the psa column
df = df[df['psa'] < 1000]

# Mean imputation for filling NAs (median imputation may be also applied if outliers)
df = df.fillna(df.median())


#No-IID division (asymmetric)
df_ordenado = df.sort_values(by='psa')
total_rows = len(df_ordenado)
corte_1 = int(total_rows * 0.80)
corte_2 = int(total_rows * 0.90)

df_client1 = df_ordenado.iloc[:corte_1]           # 80% (Large hospital)
df_client2 = df_ordenado.iloc[corte_1:corte_2]    # 10% (Small health center A)
df_client3 = df_ordenado.iloc[corte_2:]           # 10% (Small health center B)


df_client1.to_csv("data/client1.csv", sep=';', index=False)
df_client2.to_csv("data/client2.csv", sep=';', index=False)
df_client3.to_csv("data/client3.csv", sep=';', index=False)

print("Datasets were generated correctly")