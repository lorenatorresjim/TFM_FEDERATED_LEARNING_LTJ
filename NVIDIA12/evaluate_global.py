import sys
import os
sys.path.append(os.path.abspath("my_tfm_job"))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from app.custom.model import create_model  # <-- Ahora esto funcionará perfectamente


MODEL_PATH = "poc_workspace/example_project/prod_00/admin@nvidia.com/transfer/67176afe-fc00-4307-af5e-9489b429dbf1/workspace/app_server/custom/server.npy"
CLIENT_FILES = ["data/client1.csv", "data/client2.csv", "data/client3.csv"]

def load_global_weights(model, path):
    loaded_data = np.load(path, allow_pickle=True)
    
    if loaded_data.size == 1:
        data_dict = loaded_data.item()
        if isinstance(data_dict, dict):
            server_weights_flat = data_dict.get("numpy_key", list(data_dict.values())[0])
        else:
            server_weights_flat = loaded_data
    else:
        server_weights_flat = loaded_data
        
    current_weights = model.get_weights()
    new_weights = []
    idx = 0
    for w in current_weights:
        size = np.prod(w.shape)
        new_weights.append(server_weights_flat[idx:idx+size].reshape(w.shape))
        idx += size
    model.set_weights(new_weights)
    return model

print("=== MULTISITE EVALUATION ===")

all_y_true = []
all_y_pred = []

for file_path in CLIENT_FILES:
    if not os.path.exists(file_path):
        continue
        
    client_name = os.path.basename(file_path).replace(".csv", "")
    print(f"\n--- Evaluating in: {client_name} ---")
    
    df = pd.read_csv(file_path, sep=';')
    X = df.drop(columns=['case_csPCa'])
    y = df['case_csPCa']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
 
    model = create_model(X_test.shape[1])
    model = load_global_weights(model, MODEL_PATH)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Precision for {client_name}: {acc:.4f}")

print("\n" + "="*45)
print("=== COMBINED GLOBAL RESULTS ===")
print(f"Total evaluated samples: {len(all_y_true)}")
print(f"Global accuracy: {accuracy_score(all_y_true, all_y_pred):.4f}")
print("\nMetrics report:")
print(classification_report(all_y_true, all_y_pred))