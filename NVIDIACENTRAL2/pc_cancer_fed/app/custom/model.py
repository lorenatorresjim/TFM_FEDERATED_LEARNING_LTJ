import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_dim):
    model = models.Sequential([
        layers.Dense(32, activation='relu', input_dim=input_dim),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
