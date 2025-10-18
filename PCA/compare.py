
'''
Compare different PCA implementations
'''


import logging as log
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_log_data(file_path):
    """
    Carga datos de archivos .log en el directorio especificado.
    
    Args:
        file_path: Ruta al directorio con archivos .log
        
    Returns:
        dict: Diccionario {nombre_metodo: DataFrame con países y valores PC1}
    """
    log.info("Loading log data from %s", file_path)
    
    # Obtener todos los archivos .log
    files = os.listdir(file_path)
    files = [f for f in files if f.endswith('.log')]
    
    results = {}
    
    for file in files:
        log.info("Processing file: %s", file)
        file_full_path = os.path.join(file_path, file)
        method_name = os.path.splitext(file)[0]
        countries = []
        values = []
        
        with open(file_full_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ';' in line:
                    parts = line.split(';')
                    if len(parts) == 2:
                        country = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            countries.append(country)
                            values.append(value)
                        except ValueError:
                            log.warning(f"Could not parse value in line: {line}")


        if countries and values:
            df = pd.DataFrame({
                'Country': countries,
                'PC1': values
            })
            results[method_name] = df
            log.info(f"Loaded {len(df)} entries from {method_name}")
        else:
            log.warning(f"No valid data found in {file}")
    
    return results

def compare_results(results):
    """
    Compara resultados de diferentes implementaciones PCA.
    
    Args:
        results: dict {nombre_metodo: DataFrame}
        
    Returns:
        DataFrame con comparaciones
    """
    log.info("Comparing PCA implementations...")
    
    if len(results) < 2:
        log.error("Need at least 2 implementations to compare")
        return None
    
    # Obtener nombres de métodos
    methods = list(results.keys())
    log.info(f"Comparing methods: {methods}")
    
    # Usar el primer método como referencia
    reference_method = methods[0]
    df_ref = results[reference_method].copy()
    df_ref = df_ref.sort_values('Country').reset_index(drop=True)
    
    # Crear DataFrame de comparación
    comparison = df_ref[['Country']].copy()
    comparison[f'{reference_method}_PC1'] = df_ref['PC1']
    
    # Agregar otros métodos
    for method in methods[1:]:
        df_other = results[method].copy()
        df_other = df_other.sort_values('Country').reset_index(drop=True)
        
        # Verificar que los países coincidan
        if not df_ref['Country'].equals(df_other['Country']):
            log.warning(f"Countries don't match exactly between {reference_method} and {method}")
            # Hacer merge por país
            comparison = comparison.merge(
                df_other[['Country', 'PC1']], 
                on='Country', 
                how='outer', 
                suffixes=('', f'_{method}')
            )
            comparison.rename(columns={'PC1': f'{method}_PC1'}, inplace=True)
        else:
            comparison[f'{method}_PC1'] = df_other['PC1']
    
    return comparison


def analyze_differences(comparison, methods):
    """
    Analiza diferencias entre implementaciones.
    
    Args:
        comparison: DataFrame con resultados de todos los métodos
        methods: lista de nombres de métodos
    """
    log.info("\n" + "="*60)
    log.info("ANÁLISIS DE DIFERENCIAS")
    log.info("="*60)
    
    reference = methods[0]
    
    for method in methods[1:]:
        ref_col = f'{reference}_PC1'
        method_col = f'{method}_PC1'
        
        if method_col not in comparison.columns:
            log.warning(f"Column {method_col} not found")
            continue
        
        log.info(f"\n{reference} vs {method}:")
        log.info("-" * 40)
        
        # Calcular diferencias
        diff = comparison[ref_col] - comparison[method_col]
        abs_diff = np.abs(diff)
        
        # Estadísticas
        log.info(f"  Media de diferencias: {diff.mean():.6f}")
        log.info(f"  Desviación estándar: {diff.std():.6f}")
        log.info(f"  Diferencia absoluta promedio: {abs_diff.mean():.6f}")
        log.info(f"  Diferencia máxima: {abs_diff.max():.6f}")
        log.info(f"  Diferencia mínima: {abs_diff.min():.6f}")
        
        # Correlación
        corr = comparison[ref_col].corr(comparison[method_col])
        log.info(f"  Correlación: {corr:.6f}")
        
        # Países con mayor diferencia
        comparison['diff_abs'] = abs_diff
        top_diff = comparison.nlargest(5, 'diff_abs')
        
        log.info(f"\n  Top 5 países con mayor diferencia:")
        for _, row in top_diff.iterrows():
            log.info(f"    {row['Country']:20s}: {ref_col}={row[ref_col]:8.4f}, "
                    f"{method_col}={row[method_col]:8.4f}, diff={row['diff_abs']:8.4f}")
        
        comparison.drop('diff_abs', axis=1, inplace=True)
    
    log.info("\n" + "="*60)


def plot_comparison(comparison, methods, output_path='out/comparison.png'):
    """
    Genera gráficos de comparación.
    
    Args:
        comparison: DataFrame con resultados
        methods: lista de nombres de métodos
        output_path: ruta para guardar el gráfico
    """
    log.info("Generating comparison plots...")
    
    n_methods = len(methods)
    
    if n_methods == 2:
        # Gráfico de dispersión para 2 métodos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        method1_col = f'{methods[0]}_PC1'
        method2_col = f'{methods[1]}_PC1'
        
        # Scatter plot
        ax1.scatter(comparison[method1_col], comparison[method2_col], alpha=0.6)
        
        # Línea de identidad (y=x)
        min_val = min(comparison[method1_col].min(), comparison[method2_col].min())
        max_val = max(comparison[method1_col].max(), comparison[method2_col].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        
        ax1.set_xlabel(f'{methods[0]} PC1')
        ax1.set_ylabel(f'{methods[1]} PC1')
        ax1.set_title(f'Comparación: {methods[0]} vs {methods[1]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de diferencias
        diff = comparison[method1_col] - comparison[method2_col]
        comparison_sorted = comparison.sort_values(method1_col).reset_index(drop=True)
        diff_sorted = comparison_sorted[method1_col] - comparison_sorted[method2_col]
        
        ax2.bar(range(len(diff_sorted)), diff_sorted, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='-')
        ax2.set_xlabel('País (ordenado por PC1)')
        ax2.set_ylabel(f'Diferencia ({methods[0]} - {methods[1]})')
        ax2.set_title('Diferencias entre métodos')
        ax2.grid(True, alpha=0.3, axis='y')
        
    else:
        # Gráfico de barras agrupadas para múltiples métodos
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(comparison))
        width = 0.8 / n_methods
        
        for i, method in enumerate(methods):
            col = f'{method}_PC1'
            offset = (i - n_methods/2 + 0.5) * width
            ax.bar(x + offset, comparison[col], width, label=method, alpha=0.7)
        
        ax.set_xlabel('País')
        ax.set_ylabel('PC1')
        ax.set_title('Comparación de implementaciones PCA')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison['Country'], rotation=90, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    log.info(f"Comparison plot saved to {output_path}")
    plt.show()


def save_comparison_report(comparison, methods, output_path='out/comparison_report.csv'):
    """
    Guarda reporte de comparación en CSV.
    
    Args:
        comparison: DataFrame con resultados
        methods: lista de nombres de métodos
        output_path: ruta para guardar el CSV
    """
    # Calcular diferencias
    if len(methods) >= 2:
        ref_col = f'{methods[0]}_PC1'
        for method in methods[1:]:
            method_col = f'{method}_PC1'
            comparison[f'diff_{methods[0]}_vs_{method}'] = comparison[ref_col] - comparison[method_col]
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar
    comparison.to_csv(output_path, index=False)
    log.info(f"Comparison report saved to {output_path}")


if __name__ == "__main__":
    # Configuración
    log_directory = 'logs/'
    
    # Cargar datos de logs
    results = load_log_data(log_directory)
    
    if not results:
        log.error("No valid log files found!")
    elif len(results) == 1:
        log.warning("Only one implementation found. Need at least 2 to compare.")
        method_name = list(results.keys())[0]
        log.info(f"\nData from {method_name}:")
        print(results[method_name])
    else:
        # Comparar resultados
        methods = list(results.keys())
        comparison = compare_results(results)
        
        if comparison is not None:
            # Mostrar tabla de comparación
            log.info("\nCOMPARACIÓN DE RESULTADOS:")
            print("\n" + comparison.to_string())
            
            # Análisis de diferencias
            analyze_differences(comparison, methods)
            
            # Generar gráficos
            plot_comparison(comparison, methods)
            
            # Guardar reporte
            save_comparison_report(comparison, methods)
            
            log.info("\nComparison completed successfully!")