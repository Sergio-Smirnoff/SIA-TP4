import numpy as np
import pandas as pd
import logging as log
import json
import sklearn.decomposition as skld
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_csv(file_path):
    log.info("Loading data from %s", file_path)
    data = pd.read_csv(file_path, delimiter=',')
    names = data.iloc[:, 0].values  # Guardar nombres
    headers = data.columns.tolist()
    data = data.iloc[:, 1:].values  # Datos numéricos
    headers = headers[1:]  # Excluir la columna de nombres
    log.info("Data loaded with shape %s", data.shape)
    return names, data, headers

def plot_data(names, data, output_path='out/pca_result.png'):
    log.info("Plotting data.")
    
    # Crear el scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=50)
    
    # Agregar etiquetas a cada punto
    for i, name in enumerate(names):
        plt.annotate(name, 
                    (data[i, 0], data[i, 1]),
                    fontsize=9,
                    alpha=0.8,
                    xytext=(5, 5),  # Desplazamiento del texto
                    textcoords='offset points')
    
    plt.title('PCA Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    log.info("Data plotted.")



def plot_biplot(X_reducido, paises, pca, feature_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 1. Scatter de los países (puntos)
    ax.scatter(X_reducido[:, 0], X_reducido[:, 1], alpha=0.7, s=50)
    
    # 2. Etiquetas de países
    for i, pais in enumerate(paises):
        ax.annotate(pais, (X_reducido[i, 0], X_reducido[i, 1]), 
                   fontsize=8, alpha=0.7)
    
    # 3. Vectores de las variables originales (flechas)
    # Los componentes principales nos dicen cómo contribuye cada variable
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i, feature in enumerate(feature_names):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                head_width=0.1, head_length=0.1, 
                fc='cyan', ec='cyan', alpha=0.6, linewidth=2)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, 
               feature, fontsize=10, color='cyan', 
               ha='center', va='center')
    
    # Configuración
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Biplot: Valores de las Componentes Principales 1 y 2')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/biplot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot PC1
def plot_pc1_bars(X_reducido, paises):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    indices_ordenados = np.argsort(X_reducido[:, 0])
    paises_ordenados = paises[indices_ordenados]
    pc1_ordenado = X_reducido[indices_ordenados, 0]
    
    # Crear colores: azul para positivos, otro color para negativos
    colores = ['steelblue' if x >= 0 else 'coral' for x in pc1_ordenado]
    
    # Gráfico de barras
    bars = ax.bar(range(len(paises_ordenados)), pc1_ordenado, color=colores)
    
    # Configuración
    ax.set_xticks(range(len(paises_ordenados)))
    ax.set_xticklabels(paises_ordenados, rotation=90, ha='right')
    ax.set_ylabel('PC1 value')
    ax.set_title('PC1 por Country')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('out/pc1_bars.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_pca(names, data, headers, n_components=2):
    log.info("Starting PCA computation.")
    pca = skld.PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    log.info("PCA computation finished.")
    log.info("Saving weights to out/pca_weights.csv")
    loadings_df = pd.DataFrame(
        pca.components_.T,  # Transponer para que variables sean filas
        columns=['PC1', 'PC2'],
        index=headers
    )
    loadings_df.to_csv('out/pca_weights.csv')
    log.info("Weights saved.")
    plot_biplot(transformed_data, names, pca, headers)
    plot_pc1_bars(transformed_data, names)

if __name__ == "__main__":
    # Cargar configuración
    config = json.load(open('config/default_config.json'))
    input_data = config['input']['data_path']

    names, data, headers = load_csv(input_data)

    # Normalizar los datos
    data_normalized = StandardScaler().fit_transform(data)
    with open("logs/lib_data.log","w") as f:
        f.write("Lib Data:\n")
        for name, row in zip(names, data_normalized):
            f.write(f"{name};{np.array2string(row[0], precision=4)}\n")
    log.info("Original Data Shape: %s", data_normalized.shape)
    reduced_data = perform_pca(names, data_normalized, headers, n_components=2)
    log.info("PCA process completed.")
