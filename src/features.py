import os
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from datetime import datetime

# Configuración Visual Global
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_config(config_path='config.yaml'):
    """Carga el archivo de configuración YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_directory(path):
    """Crea el directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_input_data(config):
    """Carga los datos de ventas y macroeconómicos."""
    paths = config['paths']
    fe_conf = config['feature_engineering']['input']
    
    sales_path = os.path.join(paths['cleansed_data'], fe_conf['sales_file'])
    macro_path = os.path.join(paths['features_data'], fe_conf['macro_file'])
    
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"No se encontró el archivo de ventas: {sales_path}")
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"No se encontró el archivo macro: {macro_path}")
        
    df_sales = pd.read_csv(sales_path, parse_dates=['fecha'], index_col='fecha')
    df_macro = pd.read_csv(macro_path, parse_dates=['fecha'], index_col='fecha')
    
    df_sales.sort_index(inplace=True)
    df_macro.sort_index(inplace=True)
    
    return df_sales, df_macro

def add_calendar_features(df):
    """Agrega variables de calendario: Festivos, Fines de semana, Semana Santa, Feria Flores."""
    df = df.copy()
    co_holidays = holidays.Colombia(language='es')
    
    # Listas para almacenar resultados
    festivos_list = []
    findes_list = []
    santa_list = []
    
    for date in df.index:
        # Rango diario para el mes
        days_in_month = pd.date_range(start=date, periods=date.days_in_month, freq='D')
        
        # Conteo de Festivos
        festivos = sum(1 for d in days_in_month if d in co_holidays)
        festivos_list.append(festivos)
        
        # Conteo de Fines de Semana (Sábados=5, Domingos=6)
        findes = sum(1 for d in days_in_month if d.dayofweek >= 5)
        findes_list.append(findes)
        
        # Semana Santa (Presencia de Jueves o Viernes Santo)
        is_holy = any("Jueves Santo" in co_holidays.get(d, "") or "Viernes Santo" in co_holidays.get(d, "") for d in days_in_month)
        santa_list.append(1 if is_holy else 0)
        
    df['festivos_conteo'] = festivos_list
    df['fines_semana_conteo'] = findes_list
    df['es_semana_santa'] = santa_list
    
    # Feria de las Flores (Agosto)
    df['es_feria_flores'] = (df.index.month == 8).astype(int)
    
    return df

def add_structural_flags(df, config):
    """Agrega flags de COVID y expansión Retail."""
    df = df.copy()
    flags = config['feature_engineering']['structural_flags']
    
    # COVID
    covid_start = pd.to_datetime(flags['covid_impact']['start'])
    covid_end = pd.to_datetime(flags['covid_impact']['end'])
    df['flag_covid'] = 0
    df.loc[(df.index >= covid_start) & (df.index <= covid_end), 'flag_covid'] = 1
    
    # Retail
    retail_start = pd.to_datetime(flags['retail_expansion']['start'])
    df['flag_retail'] = 0
    df.loc[df.index >= retail_start, 'flag_retail'] = 1
    
    return df

def integrate_macro_data(df_sales, df_macro, config):
    """Une las ventas con las variables macro seleccionadas."""
    macro_cols = config['feature_engineering']['macro_selection']['columns']
    
    # Join
    df_final = df_sales.join(df_macro[macro_cols], how='left')
    
    # Limpieza de nulos (si existieran por desajuste de fechas)
    if df_final[macro_cols].isna().any().any():
        df_final[macro_cols] = df_final[macro_cols].ffill().bfill()
        
    return df_final

def save_feature_visualizations(df, output_dir):
    """Genera y guarda visualizaciones de las nuevas variables."""
    ensure_directory(output_dir)
    
    # 1. Ventas vs Hitos
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df['unidades'], color='black', linewidth=1.5, label='Ventas')
    ax1.set_ylabel('Unidades Vendidas')
    
    ax2 = ax1.twinx()
    ax2.fill_between(df.index, 0, df['flag_retail'], alpha=0.15, color='green', label='Retail Impact')
    ax2.set_ylabel('Flag Retail')
    
    plt.title('Ventas Mensuales vs Impacto Expansión Retail')
    plt.savefig(os.path.join(output_dir, 'sales_vs_retail_flag.png'))
    plt.close()
    
    # 2. Correlación de Features
    plt.figure(figsize=(12, 10))
    # Excluir 'es_atipico' si no tiene varianza o muchos nulos
    corr_df = df.select_dtypes(include=[np.number])
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación: Ventas y Características Exógenas')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation_matrix.png'))
    plt.close()

def convert_numpy(obj):
    """Helper para serialización JSON de tipos NumPy."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)

def run_feature_engineering_pipeline(config_path='config.yaml'):
    """Orquestador principal de la Fase 4."""
    print("🚀 Iniciando Fase 4: Feature Engineering...")
    
    # 1. Configuración
    config = load_config(config_path)
    fe_conf = config['feature_engineering']
    paths = config['paths']
    
    # 2. Carga
    df_sales, df_macro = load_input_data(config)
    
    # 3. Transformaciones
    print("📅 Generando variables de calendario...")
    df_features = add_calendar_features(df_sales)
    
    print("🚩 Implementando flags estructurales...")
    df_features = add_structural_flags(df_features, config)
    
    print("📈 Integrando variables macroeconómicas...")
    df_final = integrate_macro_data(df_features, df_macro, config)
    
    # 4. Visualización
    viz_dir = os.path.join(paths['figures'], 'feature')
    print(f"📊 Generando visualizaciones en: {viz_dir}")
    save_feature_visualizations(df_final, viz_dir)
    
    # 5. Exportación
    output_dir = paths['processed_data']
    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, fe_conf['output']['filename'])
    
    print(f"💾 Guardando dataset final en: {output_path}")
    df_final.to_parquet(output_path, engine='pyarrow')
    
    # 6. Reporte JSON
    report_path = os.path.join(paths['reports'], fe_conf['output']['report_name'])
    
    # Validaciones Finales
    null_counts = df_final.isna().sum().to_dict()
    total_nulls = sum(null_counts.values())
    
    report_data = {
        "phase": "04_Feature_Engineering",
        "timestamp": datetime.now().isoformat(),
        "input_metrics": {
            "sales_rows": len(df_sales),
            "original_columns": df_sales.columns.tolist(),
            "macro_variables_added": fe_conf['macro_selection']['columns']
        },
        "output_metrics": {
            "total_rows": len(df_final),
            "total_columns": len(df_final.columns),
            "output_file": fe_conf['output']['filename'],
            "schema_description": {col: str(dtype) for col, dtype in df_final.dtypes.items()}
        },
        "data_quality_validation": {
            "null_values_after_fe": null_counts,
            "has_nulls": total_nulls > 0,
            "total_nulls_detected": total_nulls
        },
        "features_created": {
            "calendar": fe_conf['calendar']['features'],
            "structural": ["flag_covid", "flag_retail"]
        },
        "correlations_with_target": df_final.corr()['unidades'].drop('unidades').to_dict()
    }
    
    if total_nulls > 0:
        print(f"⚠️ Alerta: Se detectaron {total_nulls} valores nulos en el dataset final.")

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, default=convert_numpy)
        
    print(f"✅ Fase 4 Completada con éxito. Reporte: {report_path}")


if __name__ == "__main__":
    run_feature_engineering_pipeline()
