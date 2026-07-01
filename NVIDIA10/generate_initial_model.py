import os
import numpy as np
import sys

# Aseguramos que Python pueda encontrar e importar tu 'model.py' desde app/custom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'my_tfm_job', 'app', 'custom')))
from model import create_model

def generate_initial_model():
    # 1. Definir la dimensión de entrada (4 columnas: patient_age, psa, psad, prostate_volume)
    input_dim = 4  
    
    print("-> Inicializando y creando el modelo de TensorFlow...")
    model = create_model(input_dim)
    
    # 2. Obtener los pesos iniciales generados por la red
    weights = model.get_weights()
    
    # 3. Aplanar y concatenar todos los pesos en un único array unidimensional (1D)
    # Esto es crucial porque tu ProstatLearner reconstruye los pesos asumiendo un array plano
    flat_weights = np.concatenate([w.flatten() for w in weights])
    
    # 4. Crear la ruta destino de manera segura: app/models/
    output_dir = os.path.join('my_tfm_job','app', 'custom')
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Guardar el archivo como server.npy
    output_path = os.path.join(output_dir, 'server.npy')
    np.save(output_path, flat_weights)
    
    print(f"--> ¡Éxito! Modelo global inicial guardado en: {output_path}")
    print(f"--> Tamaño del vector de pesos: {flat_weights.shape}")

if __name__ == "__main__":
    generate_initial_model()