
import logging as log
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

def compare_pca_methods():
    log.info("Comparing PCA methods: Oja's rule vs. sklearn PCA.")
    csvs = os.listdir('out/')
    csvs = [f for f in csvs if f.endswith('.csv')]

    # Oja's PCA
    oja_pca = []

    for csv in csvs:
        if csv.startswith('oja'):
            struct = {"df": None, "lr":None, "epochs": None}
            log.info("Loading Oja PCA results from %s", csv)
            df = pd.read_csv(os.path.join('out/', csv))
            struct["df"] = df
            parts = csv.replace('.csv', '').split('_')
            struct["lr"] = float(parts[2].replace('lr', ''))
            struct["epochs"] = int(parts[3].replace('epochs', ''))
            oja_pca.append(struct)

    # Sklearn PCA
    sklearn_pca = []
    for csv in csvs:
        if csv.startswith('pca'):
            log.info("Loading sklearn PCA results from %s", csv)
            df = pd.read_csv(os.path.join('out/', csv))
            sklearn_pca.append(df)

    paises = sklearn_pca[0].iloc[:, 0].values  # Primera columna
    feature_names = sklearn_pca[0].columns[1:]  # Nombres de variables
    X_sklearn = sklearn_pca[0].iloc[:, 1:].values      # Datos numéricos

    # Asumiendo que ambos métodos tienen el mismo número de componentes
    for oja in oja_pca:

        log.info("Plotting comparison")
        log.debug("Oja PCA params - lr: %s, epochs: %s", oja["lr"], oja["epochs"])
        log.debug("Oja PCA shape: %s", oja["df"])
        log.debug("PCA head: %s", X_sklearn[:,0])

        X_oja = oja["df"].iloc[:, 1:].values            # Datos numéricos
        comparacion_df = pd.DataFrame({
            'Country': paises,
            'OJA': X_oja[:,0],
            'sklearn': X_sklearn[:,0]
        })
        comparacion_df['Diferencia'] = abs(comparacion_df['OJA'] - comparacion_df['sklearn'])
        log.info("Saving comparison plot.")
        fig, ax = plt.subplots(figsize=(10, len(paises) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=comparacion_df.values,
            colLabels=comparacion_df.columns,
            cellLoc='center',
            loc='center'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        plt.title(f'Comparación OJA vs sklearn - lr:{oja["lr"]}, epochs:{oja["epochs"]}', pad=20, fontsize=12)
        output_png = f'out/comparacion_PC_lr{oja["lr"]}_epochs{oja["epochs"]}.png'
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Imagen guardada en: {output_png}")

    log.info("PCA comparison plots saved.")


if __name__ == "__main__":
    compare_pca_methods()