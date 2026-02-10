import os
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Configuración Visual Global
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

def load_config(config_path='config.yaml'):
    """Carga el archivo de configuración YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_directory(path):
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)

def load_data(config):
    """Carga los datos de ventas y macroeconómicos según el config."""
    paths = config['paths']
    eda_conf = config['eda']['input']
    
    # Construcción de rutas (asumiendo ejecución desde la raíz del proyecto)
    sales_path = os.path.join(paths['cleansed_data'], eda_conf['sales_file'])
    macro_path = os.path.join(paths['features_data'], eda_conf['macro_file'])
    
    # Carga Ventas
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"No se encontró el archivo de ventas: {sales_path}")
        
    df = pd.read_csv(sales_path, parse_dates=['fecha'], index_col='fecha')
    df.sort_index(inplace=True)
    df['unidades'] = df['unidades'].astype(float)
    
    # Carga Macro (Opcional)
    df_macro = None
    if os.path.exists(macro_path):
        df_macro = pd.read_csv(macro_path, parse_dates=['fecha'], index_col='fecha')
        print("✅ Datos macroeconómicos cargados exitosamente.")
    else:
        print(f"⚠️ Alerta: Archivo macro no encontrado en {macro_path}")
        
    return df, df_macro

def perform_splitting(df, config):
    """Divide el dataset en Train/Val/Test usando fechas fijas."""
    split_conf = config['eda']['splitting']
    
    train = df.loc[split_conf['train']['start_date']:split_conf['train']['end_date']].copy()
    val = df.loc[split_conf['validation']['start_date']:split_conf['validation']['end_date']].copy()
    test = df.loc[split_conf['test']['start_date']:split_conf['test']['end_date']].copy()
    
    return train, val, test

def analyze_drift(train, val, test):
    """Analiza cambios de media y distribución entre sets."""
    stats = {
        'train_mean': train['unidades'].mean(),
        'val_mean': val['unidades'].mean(),
        'test_mean': test['unidades'].mean(),
        'train_std': train['unidades'].std(),
        'val_std': val['unidades'].std()
    }
    
    # Detección de Drift
    drift_pct = ((stats['val_mean'] - stats['train_mean']) / stats['train_mean']) * 100
    stats['drift_val_vs_train_pct'] = round(drift_pct, 2)
    stats['drift_detected'] = abs(drift_pct) > 20
    
    return stats

def run_stationarity_test(series):
    """Ejecuta el test de Dickey-Fuller."""
    result = adfuller(series.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05
    }

def analyze_calendar_effects(df, output_dir):
    """Calcula y visualiza impacto de Festivos, Fines de Semana, y Eventos."""
    df_cal = df.copy()
    
    # Feature Engineering de Calendario
    co_holidays = holidays.Colombia()
    
    df_cal['n_festivos'] = df_cal.index.map(lambda x: sum([1 for d in pd.date_range(x, periods=x.days_in_month) if d in co_holidays]))
    df_cal['n_fines_semana'] = df_cal.index.map(lambda x: sum([1 for d in pd.date_range(x, periods=x.days_in_month) if d.weekday() >= 5]))
    
    # Reglas de negocio (Feria Flores = Agosto, Semana Santa ~ Marzo/Abril)
    df_cal['es_feria_flores'] = np.where(df_cal.index.month == 8, 'Si', 'No')
    df_cal['es_semana_santa'] = np.where(df_cal.index.month.isin([3, 4]), 'Posible', 'No')
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sns.boxplot(data=df_cal, x='n_festivos', y='unidades', hue='n_festivos', legend=False, ax=axes[0, 0], palette='Blues')
    axes[0, 0].set_title('Impacto: Cantidad de Festivos')
    
    sns.boxplot(data=df_cal, x='n_fines_semana', y='unidades', hue='n_fines_semana', legend=False, ax=axes[0, 1], palette='Greens')
    axes[0, 1].set_title('Impacto: Cantidad de Fines de Semana')
    
    sns.boxplot(data=df_cal, x='es_feria_flores', y='unidades', hue='es_feria_flores', legend=False, ax=axes[1, 0], palette='Oranges')
    axes[1, 0].set_title('Impacto: Feria de Flores (Agosto)')
    
    sns.boxplot(data=df_cal, x='es_semana_santa', y='unidades', hue='es_semana_santa', legend=False, ax=axes[1, 1], palette='Purples')
    axes[1, 1].set_title('Impacto: Meses de Semana Santa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calendar_impact.png'))
    plt.close()
    
    return {
        'avg_sales_feria': df_cal[df_cal['es_feria_flores']=='Si']['unidades'].mean(),
        'avg_sales_normal': df_cal[df_cal['es_feria_flores']=='No']['unidades'].mean()
    }

def run_eda_analysis(config_path='config.yaml'):
    """
    Función orquestadora principal del EDA.
    Ejecuta todo el pipeline de análisis exploratorio y genera reportes.
    """
    print("🚀 Iniciando Pipeline de EDA...")
    
    # 1. Carga
    config = load_config(config_path)
    df, df_macro = load_data(config)
    
    # Directorios de salida
    figures_dir = os.path.join(config['paths']['figures'], config['eda']['output']['figures_dir'])
    reports_dir = config['paths']['reports']
    ensure_directory(figures_dir)
    ensure_directory(reports_dir)

    # --- 0. Metadata Report Initialization ---
    report_data = {
        "phase": "03_EDA",
        "timestamp": pd.Timestamp.now().isoformat(),
        "input_metrics": {},
        "output_metrics": {},
        "dataset_stats": {
            "overall": {},
            "train": {},  # New section for train-specific stats
        },
        "analysis_results": {}
    }

    # --- 1. Carga & Input Metrics ---
    # config already loaded above
    # df, df_macro already loaded above
    
    # Directorios de salida
    figures_dir = os.path.join(config['paths']['figures'], config['eda']['output']['figures_dir'])
    reports_dir = config['paths']['reports']
    ensure_directory(figures_dir)
    ensure_directory(reports_dir)
    
    # Capture Input Metrics
    report_data['input_metrics'] = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "column_types": df.dtypes.astype(str).to_dict(),
        "date_range": {
            "start": str(df.index.min()),
            "end": str(df.index.max())
        },
        "frequency": pd.infer_freq(df.index) if len(df) > 2 else "Unknown"
    }
    
    # --- 2. Descriptive & Distribution Stats ---
    desc_stats = df['unidades'].describe().to_dict()
    report_data['dataset_stats']['overall'] = desc_stats
    
    # Monthly Stats (Seasonality check)
    monthly_stats = df.groupby(df.index.month)['unidades'].mean().to_dict()
    report_data['dataset_stats']['monthly_avg_sales'] = monthly_stats

    # --- 3. Split & Drift Analysis ---
    train, val, test = perform_splitting(df, config)
    drift_stats = analyze_drift(train, val, test)
    
    split_info = {
         "train_size": len(train),
         "val_size": len(val),
         "test_size": len(test)
    }
    report_data['analysis_results']['data_splitting'] = split_info
    report_data['analysis_results']['drift_analysis'] = drift_stats
    
    print(f"📊 Drift Check: {drift_stats['drift_detected']} (Dif: {drift_stats['drift_val_vs_train_pct']}%)")
    
    # (Visualización Split)
    plt.figure()
    plt.plot(train.index, train['unidades'], label='Train')
    plt.plot(val.index, val['unidades'], label='Validation')
    plt.plot(test.index, test['unidades'], label='Test')
    plt.legend()
    plt.title('Data Splitting')
    plt.savefig(os.path.join(figures_dir, 'splitting.png'))
    plt.close()
    
    # --- 4. Descomposición (Sólo Train) ---
    print("📉 Ejecutando Descomposición sobre TRAIN...")
    decomp = seasonal_decompose(train['unidades'], model='additive', period=12)
    decomp.plot().set_size_inches(14, 10)
    plt.savefig(os.path.join(figures_dir, 'decomposition_train.png'))
    plt.close()
    report_data['analysis_results']['decomposition'] = "Figures generated: decomposition_train.png (Calculated on Train set)"
    
    # --- 5. Estacionariedad (Sólo Train) ---
    adf_result = run_stationarity_test(train['unidades'])
    report_data['analysis_results']['stationarity_test'] = adf_result
    report_data['analysis_results']['stationarity_test']['note'] = "Calculated on Train set only"
    report_data['analysis_results']['stationarity_test'] = adf_result
    
    # --- 6. Distribución Visual (Comparativa Train vs Full pero stats sobre Train) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma solo de TRAIN para ver la distribución que aprenderá el modelo
    sns.histplot(train['unidades'], kde=True, ax=ax[0], color='blue', label='Train')
    ax[0].set_title("Distribución de Ventas (Train Set)")
    
    # Boxplot mensual sobre TRAIN para ver estacionalidad aprendible
    train_viz = train.copy()
    train_viz['mes'] = train_viz.index.month
    sns.boxplot(data=train_viz, x='mes', y='unidades', hue='mes', legend=False, ax=ax[1])
    ax[1].set_title("Estacionalidad Mensual (Train Set)")
    
    plt.savefig(os.path.join(figures_dir, 'distribution_seasonality_train.png'))
    plt.close()
    
    # Stats de TRAIN
    report_data['dataset_stats']['train'] = train['unidades'].describe().to_dict()
    
    # --- 7. Autocorrelación (Sólo Train) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    plot_acf(train['unidades'], lags=24, ax=ax[0])
    ax[0].set_title("Autocorrelation (Train)")
    plot_pacf(train['unidades'], lags=12, method='ywm', ax=ax[1])
    ax[1].set_title("Partial Autocorrelation (Train)")
    plt.savefig(os.path.join(figures_dir, 'autocorrelation_train.png'))
    plt.close()
    report_data['analysis_results']['autocorrelation'] = "Figures generated: autocorrelation_train.png (Calculated on Train set)"
    
    # --- 8. Calendario y Eventos (Sólo Train) ---
    calendar_fx = analyze_calendar_effects(train, figures_dir)
    report_data['analysis_results']['calendar_effects'] = calendar_fx
    
    # --- 9. Hitos Estructurales (Business Rules) ---
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['unidades'], color='black', alpha=0.7)
    
    rules = config['eda']['business_rules']['structural_breaks']
    # Covid
    plt.axvspan(pd.to_datetime(rules['covid_period']['start']), 
                pd.to_datetime(rules['covid_period']['end']), 
                color='red', alpha=0.2, label='COVID-19')
    # Retail
    plt.axvline(pd.to_datetime(rules['gran_superficie_contract']['start']), 
                color='green', linestyle='--', linewidth=2, label='Contrato Retail')
    
    plt.legend()
    plt.title("Hitos Estructurales")
    plt.savefig(os.path.join(figures_dir, 'structural_breaks.png'))
    plt.close()
    
    report_data['analysis_results']['structural_breaks'] = {
        "covid_period": rules['covid_period'],
        "retail_contract": rules['gran_superficie_contract'],
        "figure": "structural_breaks.png"
    }

    # --- 10. Macro (Si existe - Sobre Train) ---
    if df_macro is not None:
        # Join solo con train para correlaciones
        merged = train.join(df_macro, how='inner')
        if not merged.empty:
            plt.figure(figsize=(8, 6))
            sns.heatmap(merged.corr(), annot=True, cmap='coolwarm')
            plt.title('Correlación Macro (Train Set)')
            plt.savefig(os.path.join(figures_dir, 'macro_correlation_train.png'))
            plt.close()
            # Guardar correlaciones
            corrs = merged.corr()['unidades'].drop('unidades').to_dict()
            report_data['analysis_results']['macro_correlations'] = corrs
            report_data['analysis_results']['macro_correlations']['note'] = "Calculated on Train set only"
    
    # --- Final Output Metrics ---
    report_data['output_metrics'] = {
        "figures_generated_count": len(os.listdir(figures_dir)),
        "figures_location": figures_dir,
        "report_location": reports_dir
    }
    
    # Guardar Reporte JSON Principal (Fase 3)
    report_path = os.path.join(reports_dir, config['eda']['output']['report_name'])
    
    # Convert numpy types to native for JSON serialization
    
    # Convert numpy types to native for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
            
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, default=convert_numpy)
        
    print(f"✅ EDA Finalizado. Reporte guardado en: {report_path}")

    # --- 11. Nuevo: Guardar Métricas de Data Drift (Histórico Inmutable) ---
    metrics_dir = config['paths']['metrics']
    ensure_directory(metrics_dir)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    prefix = config['eda']['output'].get('drift_report_prefix', 'data_drift')
    drift_filename = f"{prefix}_{timestamp}.json"
    drift_path = os.path.join(metrics_dir, drift_filename)
    
    drift_metrics = {
        "data_splitting": report_data['analysis_results']['data_splitting'],
        "drift_analysis": report_data['analysis_results']['drift_analysis']
    }
    
    with open(drift_path, 'w', encoding='utf-8') as f:
        json.dump(drift_metrics, f, indent=4, default=convert_numpy)
        
    print(f"📉 Métricas de Data Drift guardadas (Histórico): {drift_path}")
    print(f"✅ Figuras guardadas en: {figures_dir}")

if __name__ == "__main__":
    run_eda_analysis()
