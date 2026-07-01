import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from model import create_model
from sklearn.utils.class_weight import compute_class_weight

class ProstateLearner(Executor):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.data_path = "/mnt/c/Users/loren/Downloads/TFM_2/NVIDIA/NVIDIA11/data"

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
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            
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
                self.logger.error(f"Error crítico al aplicar los pesos globales del servidor: {e}") 

            
            classes = np.unique(y_train)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, weights))
            
            self.model.fit(X_train, y_train, epochs=5, verbose=0, class_weight=class_weight_dict)
            
            weights = self.model.get_weights()
            flat_weights = np.concatenate([w.flatten() for w in weights])
            
            dxo = DXO(
                data_kind=DataKind.WEIGHTS, 
                data={"numpy_key": flat_weights},
                meta={"NUM_STEPS_CURRENT_ROUND": 1.0} 
            )
            return dxo.to_shareable()
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)
