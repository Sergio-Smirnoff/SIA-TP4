import numpy as np
import pandas as pd
import logging as log
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.preprocessing import StandardScaler
from kohonen import Kohonen

# log.basicConfig(
#     level=log.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s'
#     )

def load_csv(file_path):
    log.info("Loading data from %s", file_path)
    data = pd.read_csv(file_path, delimiter=',')
    names = data.iloc[:, 0].values  # Guardar nombres
    headers = data.columns.tolist()
    data = data.iloc[:, 1:].values  # Datos numéricos
    headers = headers[1:]  # Excluir la columna de nombres
    log.info("Data loaded with shape %s", data.shape)
    return names, data, headers

#======================== ANALYSIS ===================================

def heat_map(neurons, k, names, output_path):
    log.info("Generating heatmap analysis plot...")
    # 1. Contar entradas por neurona
    hit_map = np.zeros((k, k))
    for (i, j) in neurons:
        hit_map[i, j] += 1
        
    # 2. Crear un mapa de etiquetas para los países
    label_map = {}
    for t, (i, j) in enumerate(neurons):
        if (i, j) not in label_map:
            label_map[(i, j)] = []
        label_map[(i, j)].append(names[t])

    # 3. Graficar
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)  
    im = ax.imshow(hit_map, cmap='PiYG', interpolation='nearest', origin='lower')
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(np.arange(k))
    ax.set_yticklabels(np.arange(k))

    for fila in range(k):
        for col in range(k):
            count = int(hit_map[fila][col])
            if (fila, col) in label_map:
                #abreviados = [p[:3].upper() for p in label_map[(fila, col)]]
                abreviados = [p for p in label_map[(fila, col)]]
                texto = "\n".join(abreviados)
                texto += "\n"
            else:
                texto = ""
            
            texto += f"{count}"  # Mostrar siempre el número
            ax.text(col, fila, texto,
                    ha="center", va="center", color="black", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))

    ax.set_title("SOM de Países Europeos")
    fig.colorbar(im, ax=ax, label="Países por neurona")
    plt.tight_layout()
    plt.savefig(f"{output_path}/kohonen_country_heat_map.png")
    log.info("Country heat map saved")
    plt.show()


def unified_distance_matrix(u_matrix, k, output_path):
    log.info("Generating U-Matrix plot...")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    im = ax.imshow(u_matrix, cmap='RdPu', interpolation='nearest', origin='lower')

    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(np.arange(k))
    ax.set_yticklabels(np.arange(k))

    ax.set_title("Distancia entre neuronas vecinas")
    fig.colorbar(im, ax=ax, label="Distancia promedio")
    plt.tight_layout()
    plt.savefig(f"{output_path}/kohonen_u_matrix.png")
    log.info("U-Matrix saved")
    plt.show()

#=====================================================================

if __name__ == "__main__":
    np.random.seed(56)
    # Cargar configuración
    config = json.load(open('config/kohonen_config.json'))
    input_data = config['input']['data_path']
    output_path = config['output']['results_path']

    names, data, headers = load_csv(input_data)
    
    # Normalizar los datos
    data_normalized = StandardScaler().fit_transform(data)

    # usar Kohonen
    log.info("Starting Kohonen computation.")
    kohonen = Kohonen(
        k=config['pca_parameters']['k'],
        data=data_normalized,
        init_r=config['pca_parameters']['init_r'],
        init_learning_rate=config['pca_parameters']['init_learning_rate'], 
        epochs_rate=config['pca_parameters']['epochs_rate'],
        similarity=config['pca_parameters']['similarity'],
        replace=config['pca_parameters']['replace']
    )
    kohonen.train()

    neurons, u_matrix = kohonen.predict(data_normalized)

    heat_map(neurons, kohonen.k, names, output_path)

    unified_distance_matrix(u_matrix, kohonen.k, output_path)
    