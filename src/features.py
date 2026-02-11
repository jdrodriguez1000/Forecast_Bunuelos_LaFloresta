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
    """Agrega variables de calendario: Festivos, Fines de semana, Semana Santa, Feria Flores, Novenas, Puentes, Quincena."""
    df = df.copy()
    co_holidays = holidays.Colombia(language='es')
    
    # Listas para almacenar resultados
    festivos_conteo = []
    findes_conteo = []
    santa_list = []
    novenas_list = []
    puentes_list = []
    quincena_list = []
    
    for date in df.index:
        # Rango diario para el mes
        days_in_month = pd.date_range(start=date, periods=date.days_in_month, freq='D')
        
        # Conteo de Festivos
        festivos = [d for d in days_in_month if d in co_holidays]
        festivos_conteo.append(len(festivos))
        
        # Conteo de Fines de Semana (Sábados=5, Domingos=6)
        findes = [d for d in days_in_month if d.dayofweek >= 5]
        findes_conteo.append(len(findes))
        
        # Semana Santa (Presencia de Jueves o Viernes Santo)
        is_holy = any("Jueves Santo" in co_holidays.get(d, "") or "Viernes Santo" in co_holidays.get(d, "") for d in days_in_month)
        santa_list.append(1 if is_holy else 0)
        
        # Conteo de Novenas (16 al 24 de Diciembre)
        if date.month == 12:
            novenas = sum(1 for d in days_in_month if 16 <= d.day <= 24)
        else:
            novenas = 0
        novenas_list.append(novenas)
            
        # Es Puente Festivo (Lunes festivos)
        puentes = sum(1 for d in festivos if d.dayofweek == 0)
        puentes_list.append(puentes)
        
        # Efecto Quincena (Fines de semana cerca del 15 o 30)
        # Definición: Sábado o Domingo entre [13-17] o [28-31]
        quincenas = sum(1 for d in findes if (13 <= d.day <= 17) or (28 <= d.day <= 31))
        quincena_list.append(quincenas)
        
    df['festivos_conteo'] = festivos_conteo
    df['fines_semana_conteo'] = findes_conteo
    df['es_semana_santa'] = santa_list
    df['conteo_novenas'] = novenas_list
    df['es_puente_festivo'] = puentes_list
    df['efecto_quincena'] = quincena_list
    
    # Feria de las Flores (Agosto)
    df['es_feria_flores'] = (df.index.month == 8).astype(int)
    
    # Mes de Prima (Junio y Diciembre)
    df['mes_de_prima'] = df.index.month.isin([6, 12]).astype(int)
    
    # Variables de calendario estándar
    df['mes'] = df.index.month
    df['trimestre'] = df.index.quarter
    df['anio'] = df.index.year
    df['semestre'] = np.where(df.index.month <= 6, 1, 2)
    
    return df

def add_cyclic_features(df):
    """Genera variables cíclicas (sin/cos) para mes y trimestre."""
    df = df.copy()
    
    # Mes (1-12)
    df['mes_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Trimestre (1-4)
    df['trimestre_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
    df['trimestre_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)
    
    return df

def add_structural_flags(df, config):
    """Agrega flags de COVID, expansión Retail y Maduración del contrato."""
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
    
    # Maduración del Contrato
    ref_date = pd.to_datetime(flags['contract_maturation']['reference_date'])
    df['maduracion_contrato'] = ((df.index.year - ref_date.year) * 12 + 
                                (df.index.month - ref_date.month))
    df['maduracion_contrato'] = df['maduracion_contrato'].clip(lower=0)
    
    return df

def add_intensity_features(df, config):
    """Aplica el mapeo de intensidad estacional (Super Pico, Pico Alto, etc.)."""
    df = df.copy()
    intensity = config['feature_engineering']['intensity_mapping']
    
    df['intensidad_comercial'] = 0  # Default Normal
    df.loc[df.index.month.isin(intensity['super_pico']), 'intensidad_comercial'] = 3
    df.loc[df.index.month.isin(intensity['pico_alto']), 'intensidad_comercial'] = 2
    df.loc[df.index.month.isin(intensity['pico_medio']), 'intensidad_comercial'] = 1
    
    return df

def add_momentum_features(df, config):
    """Calcula crecimiento YoY y momentum (medias móviles)."""
    df = df.copy()
    fe_conf = config['feature_engineering']
    
    if fe_conf['growth_momentum']['yoy_comparison']:
        # Crecimiento porcentual contra el mismo mes del año anterior
        df['crecimiento_yoy'] = df['unidades'].pct_change(12)
        
        # Media móvil del crecimiento (Momentum)
        for w in fe_conf['growth_momentum']['rolling_windows']:
            df[f'growth_momentum_{w}m'] = df['crecimiento_yoy'].rolling(window=w).mean()
            
    return df

def impute_momentum_nulls(df):
    """Imputa los nulos iniciales de las variables de momentum usando Backfill."""
    df = df.copy()
    momentum_cols = [c for c in ['crecimiento_yoy', 'growth_momentum_3m', 'growth_momentum_6m'] if c in df.columns]
    
    if momentum_cols:
        df[momentum_cols] = df[momentum_cols].bfill()
        
    return df

def integrate_macro_data(df_sales, df_macro, config):
    """Une las ventas con las variables macro seleccionadas, manejando faltantes con Proxys."""
    macro_cols = config['feature_engineering']['macro_selection']['columns']
    
    # 1. Identificar columnas macro que existen en el archivo
    existing_in_macro = [c for c in macro_cols if c in df_macro.columns]
    
    # 2. Manejo de Proxys (Opcional según lógica previa)
    missing_macro = [c for c in macro_cols if c not in df_macro.columns]
    if 'ipc_alimentos_mensual' in missing_macro and 'ipc_mensual' in df_macro.columns:
        print("ℹ️ Usando 'ipc_mensual' como proxy para 'ipc_alimentos_mensual' en src.features.")
        df_macro['ipc_alimentos_mensual'] = df_macro['ipc_mensual']
        existing_in_macro.append('ipc_alimentos_mensual')
    
    # 3. EVITAR OVERLAP: Filtrar columnas que ya existan en df_sales (ej. 'mes_de_prima')
    cols_to_join = [c for c in existing_in_macro if c not in df_sales.columns]
    
    # Informar si hay solapamiento
    overlaps = [c for c in existing_in_macro if c in df_sales.columns]
    if overlaps:
        print(f"ℹ️ Las siguientes variables ya existen en el dataset y se omitieron del join macro: {overlaps}")

    # 4. Realizar el Join
    if cols_to_join:
        df_final = df_sales.join(df_macro[cols_to_join], how='left')
    else:
        df_final = df_sales.copy()
    
    # 5. Limpieza de nulos macro (ffill/bfill) en las columnas integradas
    if cols_to_join and df_final[cols_to_join].isna().any().any():
        df_final[cols_to_join] = df_final[cols_to_join].ffill().bfill()
        
    return df_final


def save_feature_visualizations(df, output_dir):
    """Genera y guarda visualizaciones de las nuevas variables."""
    ensure_directory(output_dir)
    
    # 1. Ventas vs Hitos y Maduración
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df.index, df['unidades'], color='black', linewidth=1.5, label='Ventas')
    ax1.set_ylabel('Unidades Vendidas')
    
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['maduracion_contrato'], color='blue', alpha=0.3, label='Maduración Contrato')
    ax2.fill_between(df.index, 0, df['flag_retail'], alpha=0.1, color='green', label='Retail Flag')
    ax2.set_ylabel('Nivel / Meses')
    
    plt.title('Ventas Mensuales vs Maduración de Contrato y Expansión Retail')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'sales_vs_maturation.png'))
    plt.close()
    
    # 2. Correlación de Features
    plt.figure(figsize=(15, 12))
    corr_df = df.select_dtypes(include=[np.number])
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación: Variables Exógenas')
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
    print("📅 Generando variables de calendario (extendidas)...")
    df_features = add_calendar_features(df_sales)
    
    print("🔄 Generando variables cíclicas (Sin/Cos)...")
    df_features = add_cyclic_features(df_features)
    
    print("🚩 Implementando flags estructurales y maduración...")
    df_features = add_structural_flags(df_features, config)
    
    print("🔥 Mapeando intensidad estacional...")
    df_features = add_intensity_features(df_features, config)
    
    print("📊 Calculando momentum y crecimiento YoY...")
    df_features = add_momentum_features(df_features, config)
    
    print("📈 Integrando variables macroeconómicas...")
    df_final = integrate_macro_data(df_features, df_macro, config)
    
    print("🧼 Imputando nulos de momentum (Backfill)...")
    df_final = impute_momentum_nulls(df_final)
    
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
    
    # 6. Reporte JSON Detallado
    report_path = os.path.join(paths['reports'], fe_conf['output']['report_name'])
    
    # Información detallada de columnas
    column_info = {}
    for col in df_final.columns:
        column_info[col] = {
            "type": str(df_final[col].dtype),
            "has_nulls": bool(df_final[col].isna().any()),
            "null_count": int(df_final[col].isna().sum())
        }
    
    report_data = {
        "phase": "04_Feature_Engineering",
        "timestamp": datetime.now().isoformat(),
        "final_rows": len(df_final),
        "final_columns_count": len(df_final.columns),
        "column_details": column_info,
        "features_created": {
            "calendar": fe_conf['calendar']['features'],
            "structural": ["flag_covid", "flag_retail", "maduracion_contrato"],
            "intensity": ["intensidad_comercial"],
            "momentum": ["crecimiento_yoy", "growth_momentum_3m", "growth_momentum_6m"]
        },
        "target_correlations": df_final.corr()['unidades'].drop('unidades').to_dict()
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, default=convert_numpy)
        
    print(f"✅ Fase 4 Completada con éxito. Reporte detallado: {report_path}")


if __name__ == "__main__":
    run_feature_engineering_pipeline()
