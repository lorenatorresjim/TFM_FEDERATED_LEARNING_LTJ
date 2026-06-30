import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from model import create_model
import sys

client_id = sys.argv[1]
df = pd.read_csv(f"data/client{client_id}.csv", sep=';')

X = df.drop(columns=['case_csPCa'])
y = df['case_csPCa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = create_model(X_train.shape[1])

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, _ = model.evaluate(X_test, y_test, verbose=0)

        # 1. Generamos predicciones locales
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 2. Calculamos accuracy y precision local
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        
        # 3. Extraemos valores de la matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

        # 4. Enviamos las métricas al servidor
        return loss, len(X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }

# Aquí se mantiene tu IP privada para la versión descentralizada
fl.client.start_numpy_client(server_address="192.168.1.43:8080", client=FlowerClient())