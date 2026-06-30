import pandas as pd
import numpy as np

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


#No-IID division
NUM_CLIENTES = 10  # Change the number based on the experiment

df_ordenado = df.sort_values(by='psa')
datasets_clientes = np.array_split(df_ordenado, NUM_CLIENTES)

for i, df_cliente in enumerate(datasets_clientes):
    cliente_id = i + 1
    df_cliente.to_csv(f"data/client{cliente_id}.csv", sep=';', index=False)

print("Datasets were generated correctly")