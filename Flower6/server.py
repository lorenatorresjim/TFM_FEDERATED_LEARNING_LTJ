import flwr as fl
import time
from typing import List, Tuple
from flwr.common import Metrics
from sklearn.metrics import classification_report

# Variables globales para controlar los rounds
CURRENT_ROUND = 1
TOTAL_ROUNDS = 3

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global CURRENT_ROUND

    # 1. Extraer los datos agregados
    examples = [num_examples for num_examples, _ in metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]

    # 2. Calcular las medias ponderadas
    agg_accuracy = sum(accuracies) / sum(examples)
    agg_precision = sum(precisions) / sum(examples)

    # 3. Sumar la matriz de confusión de toda la red
    tp = sum([m["tp"] for _, m in metrics])
    tn = sum([m["tn"] for _, m in metrics])
    fp = sum([m["fp"] for _, m in metrics])
    fn = sum([m["fn"] for _, m in metrics])

    print(f"\n{'='*40}")
    print(f"--- RESULTADOS DEL ROUND {CURRENT_ROUND} ---")
    print(f"Global Accuracy:  {agg_accuracy:.4f}")
    print(f"Global Precision: {agg_precision:.4f}")
    print(f"{'='*40}\n")

    # 4. Imprimir el classification report si es el último round
    if CURRENT_ROUND == TOTAL_ROUNDS:
        print("\n" + "*"*55)
        print(" FINAL GLOBAL CLASSIFICATION REPORT ")
        print("*"*55)

        y_true = [1]*int(tp) + [1]*int(fn) + [0]*int(tn) + [0]*int(fp)
        y_pred = [1]*int(tp) + [0]*int(fn) + [0]*int(tn) + [1]*int(fp)

        report = classification_report(y_true, y_pred, digits=4)
        print(report)

    CURRENT_ROUND += 1

    return {"accuracy": agg_accuracy, "precision": agg_precision}

# Configuración de la estrategia agregando la función de evaluación
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average, 
)

# Inicio del servidor con el temporizador y configuración 0.0.0.0
start_time = time.time()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
)

end_time = time.time()
print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")