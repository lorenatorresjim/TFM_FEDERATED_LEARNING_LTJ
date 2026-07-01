import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report # IMPORTACIONES DE SKLEARN
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from model import create_model
from sklearn.utils.class_weight import compute_class_weight

# Importaciones necesarias para el agregador que se ejecutará en el servidor
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.apis.fl_context import FLContext

# =====================================================================
# INTERFAZ DEL CLIENTE (EJECUTADA POR LOS CLIENTES)
# =====================================================================
class ProstateLearner(Executor):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.data_path = "/mnt/c/Users/loren/Downloads/TFM_2/NVIDIA/NVIDIA/data"

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if task_name == "train":
            client_name = fl_ctx.get_identity_name()
            cid = re.findall(r'\d+', client_name)[0] if re.findall(r'\d+', client_name) else "1"
            csv_path = os.path.join(self.data_path, f"client{cid}.csv")
            
            if not os.path.exists(csv_path):
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            df = pd.read_csv(csv_path, sep=';')
            X = df.drop(columns=['case_csPCa'])
            y = df['case_csPCa']
            
            # Split 80/20 para evaluar localmente en cada round con el modelo global
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            if self.model is None:
                self.model = create_model(X_train.shape[1])
            
            try:
                dxo_in = from_shareable(shareable)
                global_weights = dxo_in.data
                if global_weights and "numpy_key" in global_weights:
                    server_weights_flat = global_weights["numpy_key"]
          
                    current_weights = self.model.get_weights()
                    new_weights = []
                    idx = 0
                    for w in current_weights:
                        size = np.prod(w.shape)
                        reshaped_w = server_weights_flat[idx:idx+size].reshape(w.shape)
                        new_weights.append(reshaped_w)
                        idx += size
                        
                    self.model.set_weights(new_weights)
            except Exception as e:
                self.logger.error(f"Error al aplicar los pesos globales: {e}") 

            # --- EVALUACIÓN DEL MODELO GLOBAL RECIBIDO DEL SERVIDOR ---
            y_pred_prob = self.model.predict(X_val, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0,1]).ravel()

            metrics = {
                "num_examples": len(y_val),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
            
            # Entrenamiento local
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, weights))
            
            self.model.fit(X_train, y_train, epochs=5, verbose=0, class_weight=class_weight_dict)
            
            weights = self.model.get_weights()
            flat_weights = np.concatenate([w.flatten() for w in weights])
            
            dxo = DXO(
                data_kind=DataKind.WEIGHTS, 
                data={"numpy_key": flat_weights},
                meta={
                    "NUM_STEPS_CURRENT_ROUND": 1.0,
                    "metrics": metrics # Enviamos las métricas en los metadatos
                }
            )
            return dxo.to_shareable()
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)


# =====================================================================
# LOGICA DEL AGREGADOR (EJECUTADA POR EL SERVIDOR)
# =====================================================================
class MetricsAggregator(InTimeAccumulateWeightedAggregator):
    def __init__(self, expected_data_kind="WEIGHTS", total_rounds=5):
        super().__init__(expected_data_kind=expected_data_kind)
        self.total_rounds = total_rounds
        self.metrics_buffer = []
        self.current_round = 1

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        accepted = super().accept(shareable, fl_ctx)
        if accepted:
            try:
                dxo = from_shareable(shareable)
                if dxo.meta and "metrics" in dxo.meta:
                    self.metrics_buffer.append(dxo.meta["metrics"])
            except Exception as e:
                self.log_error(fl_ctx, f"Error al extraer métricas: {e}")
        return accepted

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        aggr_shareable = super().aggregate(fl_ctx)
        
        if self.metrics_buffer:
            examples = [m["num_examples"] for m in self.metrics_buffer]
            accuracies = [m["num_examples"] * m["accuracy"] for m in self.metrics_buffer]
            precisions = [m["num_examples"] * m["precision"] for m in self.metrics_buffer]

            agg_accuracy = sum(accuracies) / sum(examples)
            agg_precision = sum(precisions) / sum(examples)

            tp = sum([m["tp"] for m in self.metrics_buffer])
            tn = sum([m["tn"] for m in self.metrics_buffer])
            fp = sum([m["fp"] for m in self.metrics_buffer])
            fn = sum([m["fn"] for m in self.metrics_buffer])

            print(f"\n{'='*40}")
            print(f"--- RESULTADOS DEL ROUND {self.current_round} ---")
            print(f"Global Accuracy:  {agg_accuracy:.4f}")
            print(f"Global Precision: {agg_precision:.4f}")
            print(f"{'='*40}\n")

            if self.current_round == self.total_rounds:
                print("\n" + "*"*55)
                print(" FINAL GLOBAL CLASSIFICATION REPORT ")
                print("*"*55)
                y_true = [1]*int(tp) + [1]*int(fn) + [0]*int(tn) + [0]*int(fp)
                y_pred = [1]*int(tp) + [0]*int(fn) + [0]*int(tn) + [1]*int(fp)
                
                print(classification_report(y_true, y_pred, digits=4))

            self.current_round += 1
            self.metrics_buffer = []

        return aggr_shareable