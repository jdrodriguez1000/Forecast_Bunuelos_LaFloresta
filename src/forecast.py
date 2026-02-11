import os
import sys
import yaml
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Suprimir avisos de librerías externas para una salida limpia en producción
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Añadir el directorio raíz al path para importaciones locales
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (
    add_calendar_features, 
    add_cyclic_features, 
    add_intensity_features, 
    add_structural_flags
)

def load_config(config_path='config.yaml'):
    """Carga el archivo de configuración YAML."""
    if not os.path.exists(config_path):
        # Intentar ruta relativa si se llama desde src/
        config_path = os.path.join('..', config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_forecast(df, config, steps=6):
    """
    Genera el pronóstico de ventas para los próximos N meses usando el modelo campeón.
    Implementa la proyección recursiva de variables macroeconómicas y cuantificación de incertidumbre.
    """
    # 1. Cargar el modelo campeón
    model_dir = config['paths'].get('models', 'outputs/models/')
    model_path = os.path.join(model_dir, 'final_model.joblib')
    
    if not os.path.exists(model_path):
        if not os.path.exists(model_path):
            parent_model_path = os.path.join('..', model_path)
            if os.path.exists(parent_model_path):
                model_path = parent_model_path
            
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo final en: {model_path}")
        
    print(f"📦 Cargando modelo: {model_path}")
    champion_forecaster = joblib.load(model_path)
    
    # Obtener variables exógenas esperadas por el modelo
    expected_exog = []
    if hasattr(champion_forecaster, 'exog_col_names'):
        expected_exog = champion_forecaster.exog_col_names
    else:
        try:
            importance = champion_forecaster.get_feature_importances(step=1)
            if 'feature' in importance.columns:
                all_features = importance['feature'].tolist()
            else:
                all_features = importance.index.tolist()
            expected_exog = [f for f in all_features if not (f.startswith('lag_') or f.startswith('roll_'))]
        except:
            expected_exog = None
    
    # 2. Preparar fechas futuras
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    exog_future = pd.DataFrame(index=future_dates)
    
    # 3. Generar variables base
    exog_future = add_calendar_features(exog_future)
    exog_future = add_cyclic_features(exog_future)
    exog_future = add_intensity_features(exog_future, config)
    exog_future = add_structural_flags(exog_future, config)
    
    # 4. Lógica de Proyección Recursiva (Económicas y Técnicas)
    cols_recursive = [
        'ipc_mensual', 'trm_promedio', 'tasa_desempleo', 'costo_insumos_index', 
        'ipc_alimentos_mensual', 'crecimiento_yoy', 'growth_momentum_3m', 'growth_momentum_6m'
    ]
    cols_to_process = [c for c in cols_recursive if c in df.columns]

    print(f"🔄 Proyectando recursivamente {len(cols_to_process)} variables...")
    for i, date in enumerate(exog_future.index):
        for col in cols_to_process:
            if i == 0:
                val = df[col].iloc[-2:].mean()
            elif i == 1:
                val = (df[col].iloc[-1] + exog_future.loc[exog_future.index[0], col]) / 2
            else:
                val = (exog_future.loc[exog_future.index[i-1], col] + exog_future.loc[exog_future.index[i-2], col]) / 2
            exog_future.loc[date, col] = val

    # 5. Ajustes finales de Reglas de Negocio
    exog_future['mes_de_prima'] = exog_future.index.month.isin([6, 12]).astype(int)
    if 'es_atipico' not in exog_future.columns:
        exog_future['es_atipico'] = 0

    # 6. Alinear exógenas con lo que espera el modelo
    if expected_exog:
        for col in expected_exog:
            if col not in exog_future.columns:
                exog_future[col] = 0
        exog_future_aligned = exog_future[expected_exog]
    else:
        exog_future_aligned = exog_future

    # 🛡️ 7. Cuantificación de la Incertidumbre (Intervalos de Predicción)
    print("🛡️ Calculando Escenarios de Incerteza (Intervalo 95%)...")
    rs = config['forecasting'].get('random_state', 42)
    
    # Re-entrenar con almacenamiento de residuos para poder calcular intervalos
    if expected_exog:
        champion_forecaster.fit(
            y=df['unidades'], 
            exog=df[expected_exog], 
            store_in_sample_residuals=True
        )
    else:
        champion_forecaster.fit(
            y=df['unidades'], 
            store_in_sample_residuals=True
        )

    # Generar predicción con intervalos
    predictions_interval = champion_forecaster.predict_interval(
        steps=steps, 
        exog=exog_future_aligned, 
        interval=0.95,
        n_boot=250,
        random_state=rs
    )
    
    return predictions_interval, champion_forecaster, exog_future_aligned

def run_forecast_pipeline():
    """Ejecuta el flujo completo y guarda artefactos JSON/CSV/PNG."""
    print("🚀 Iniciando Motor de Pronóstico Profesional con Análisis de Influencia...")
    
    config = load_config()
    processed_path = os.path.join(
        config['paths']['processed_data'], 
        config['feature_engineering']['output']['filename']
    )
    
    if not os.path.exists(processed_path):
        processed_path = os.path.join('..', processed_path)
        
    df = pd.read_parquet(processed_path)
    
    # Asegurar frecuencia para skforecast
    if df.index.freq is None:
        df.index.freq = 'MS'
    
    # Generar pronóstico e intervalos
    predictions_interval, model, exog_future = generate_forecast(df, config)
    
    # --- Gestión de Archivos de Salida ---
    now = datetime.now()
    timestamp_str = now.strftime('%Y%m%d_%H%M%S')
    
    forecast_dir = config['paths'].get('forecasts', 'outputs/forecasts/')
    figures_dir = os.path.join(config['paths'].get('figures', 'outputs/figures/'), 'forecasts')
    
    for d in [forecast_dir, figures_dir]:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            
    # 1. Formatear para Negocio
    interval_df = predictions_interval.copy()
    mapping = {
        'pred': 'Pronóstico Puntual',
        'lower_bound': 'Escenario Pesimista (2.5%)',
        'upper_bound': 'Escenario Optimista (97.5%)'
    }
    interval_df = interval_df.rename(columns=mapping)
    interval_df.index.name = 'Mes'
    
    # 2. Visualización Principal (Pronóstico + Incertidumbre)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-12:], df['unidades'].iloc[-12:], label='Historia Reciente', color='black', marker='o', alpha=0.3)
    plt.plot(predictions_interval.index, predictions_interval['pred'], 
             label='Pronóstico Puntual', color='darkorange', marker='s', linewidth=2)
    plt.fill_between(
        predictions_interval.index,
        predictions_interval['lower_bound'], 
        predictions_interval['upper_bound'], 
        color='darkorange', alpha=0.2, label='Intervalo de Confianza (95%)'
    )
    plt.title("Pronostico de Ventas con Cuantificacion de Incertidumbre")
    plt.ylabel("Unidades Vendidas")
    plt.xlabel("Mes")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    uncertainty_plot_path = os.path.join(figures_dir, 'forecast_uncertainty_intervals.png')
    plt.savefig(uncertainty_plot_path, bbox_inches='tight')
    plt.close()

    # 3. 🔍 ANÁLISIS DE INFLUENCIA (Feature Importance)
    print("🔍 Analizando los principales impulsores del pronóstico...")
    try:
        importance = model.get_feature_importances(step=1)
        importance = importance.sort_values('importance', ascending=False)

        # Gráfica de Barras Top 20
        plt.figure(figsize=(12, 8))
        top_n = 20
        sns.barplot(
            x='importance', 
            y='feature', 
            data=importance.head(top_n), 
            hue='feature',
            palette='viridis',
            legend=False
        )
        
        # Obtener nombre del modelo para el título
        try:
            # skforecast > 0.13 usa 'estimator', versiones previas usan 'regressor'
            if hasattr(model, 'estimator'):
                reg = model.estimator
            else:
                reg = model.regressor
            model_name = str(type(reg)).split('.')[-1].replace("'>", "")
        except:
            model_name = "Campeón"

        plt.title(f"Top {top_n} Factores que Impulsan las Ventas (Modelo: {model_name})")
        plt.xlabel("Peso / Importancia en el Modelo")
        plt.ylabel("Variable")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        importance_plot_path = os.path.join(figures_dir, 'feature_importance_forecast.png')
        plt.savefig(importance_plot_path, bbox_inches='tight')
        plt.close()
        
        # Tabla de Importancia
        importance_display = importance.head(15).copy()
        if 'Ridge' in model_name:
            def interpret_impact(val):
                if val > 0: return "📈 Impulsa el aumento"
                elif val < 0: return "📉 Reduce la demanda"
                else: return "⚪ Neutral"
            importance_display['Efecto'] = importance_display['importance'].apply(interpret_impact)
        
        print("\n📋 CLASIFICACIÓN TÉCNICA DE VARIABLES (Impulsores de Demanda)")
        print(importance_display)
        print(f"\n💡 INSIGHT CLAVE: La variable '{importance.iloc[0]['feature']}' es el determinante principal.")
        
    except Exception as e:
        print(f"⚠️ No se pudo extraer la importancia detallada: {e}")

    # 4. 📈 PROYECCIÓN DE VARIABLES EXÓGENAS (Heatmap)
    print("📊 Generando visualización de proyección de variables exógenas...")
    try:
        exog_viz = exog_future.T
        plt.figure(figsize=(14, 10))
        sns.heatmap(exog_viz, annot=True, fmt=".3f", cmap='YlGnBu', cbar_kws={'label': 'Nivel Escalar'})
        plt.title("Proyeccion de Variables Exogenas (Y: Variable | X: Mes)")
        plt.xlabel("Mes de Pronóstico")
        plt.ylabel("Variables de Entrada")
        
        exog_plot_path = os.path.join(figures_dir, 'exogenous_projection.png')
        plt.savefig(exog_plot_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Mapa de calor exógeno guardado en: {exog_plot_path}")
        
        # Guardar tabla de exógenas para auditoría
        exog_csv_path = os.path.join(forecast_dir, "exogenous_projections_debug.csv")
        exog_viz.to_csv(exog_csv_path)
        
        print("\n📋 VALORES PROYECTADOS DE VARIABLES EXÓGENAS (Detalle)")
        # Formatear el DataFrame para visualización (Transpuesto para que las variables sean filas)
        # Usamos .to_string() para asegurar que pandas no lo trunque en consola
        print(exog_viz.map(lambda x: f"{x:,.4f}" if isinstance(x, (float, int)) else x).to_string())
        print(f"\n💾 Tabla completa guardada en: {exog_csv_path}")
        
    except Exception as e:
        print(f"⚠️ No se pudo generar el heatmap de exógenas: {e}")

    # 5. Exportación de Datos
    csv_path = os.path.join(forecast_dir, f"final_forecast_intervals_{timestamp_str}.csv")
    interval_df.to_csv(csv_path)

    json_path = os.path.join(forecast_dir, "final_forecast_intervals_latest.json")
    
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return str(obj)

    json_output = {
        "metadata": {
            "fecha_generacion": now.strftime('%Y-%m-%d %H:%M:%S'),
            "modelo_utilizado": model_name if 'model_name' in locals() else "Unknown",
            "confianza": "95%",
            "metodo": "Bootstrapping de residuos"
        },
        "pronostico": interval_df.reset_index().to_dict(orient='records'),
        "top_features": importance.head(10).to_dict(orient='records') if 'importance' in locals() else []
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False, default=convert_numpy)
    
    print("\n" + "="*45)
    print("🛡️ ESCENARIOS OPERATIVOS FINALES")
    print("="*45)
    print(interval_df.map(lambda x: f"{x:,.0f}"))
    print(f"\n✅ Proceso completado exitosamente.")
    print(f"📂 Artefactos generados en: {forecast_dir} y {figures_dir}")

if __name__ == '__main__':
    run_forecast_pipeline()
