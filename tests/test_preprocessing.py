import pytest
import pandas as pd
import numpy as np
import os
import yaml
import shutil
from datetime import datetime, timedelta
from src.preprocessing import DataPreprocessor

# Configuración de prueba
TEST_CONFIG = {
    "project_name": "Test Project",
    "paths": {
        "raw_data": "tests/test_data/raw_ventas.csv",
        "cleansed_data": "tests/test_data/cleansed/",
        "reports": "tests/test_data/reports/"
    },
    "preprocessing": {
        "rename_columns": {
            "unidades_vendidas_mensuales": "unidades"
        },
        "sentinels": {
            "values": [0, -1, 999],
            "action": "validate_and_nan"
        },
        "time_series": {
            "frequency": "MS",
            "exclude_current_month": True,
            "date_column": "fecha"
        },
        "imputation": {
            "method": "ffill",
            "fallback_method": "bfill"
        },
        "outliers": {
            "method": "iqr",
            "iqr_multiplier": 1.5,
            "flag_column": "es_atipico"
        },
        "output": {
            "filename": "test_clean_data.csv",
            "report_name": "test_report.json"
        }
    }
}

@pytest.fixture
def setup_test_environment():
    """Crea un entorno de prueba con archivos temporales y configuración."""
    os.makedirs("tests/test_data/cleansed", exist_ok=True)
    os.makedirs("tests/test_data/reports", exist_ok=True)
    
    # Guardar config temporal
    with open("tests/test_config.yaml", "w") as f:
        yaml.dump(TEST_CONFIG, f)
        
    # Crear dataset sintético (con huecos, sentinelas y outliers)
    dates = [
        "2023-01-01", "2023-02-01", "2023-03-01", 
        "2023-05-01",  # Falta Abril (Hueco)
        "2023-06-01", "2023-07-01", "2023-08-01"
    ]
    values = [100, 105, 0, 110, 999, 5000, 102] # 0 y 999 son sentinelas, 5000 es outlier
    
    df = pd.DataFrame({
        "fecha": dates,
        "unidades_vendidas_mensuales": values
    })
    
    df.to_csv(TEST_CONFIG["paths"]["raw_data"], index=False)
    
    yield "tests/test_config.yaml"
    
    # Cleanup
    # shutil.rmtree("tests/test_data")
    # os.remove("tests/test_config.yaml")

def test_pipeline_execution(setup_test_environment):
    """Prueba la ejecución completa del pipeline."""
    config_path = setup_test_environment
    processor = DataPreprocessor(config_path=config_path)
    
    # Ejecutar pipeline
    df_clean = processor.run_pipeline()
    
    # 1. Validar Renombramiento
    assert "unidades" in df_clean.columns
    assert "unidades_vendidas_mensuales" not in df_clean.columns
    
    # 2. Validar Limpieza de Sentinelas e Imputación
    # El valor 0 (Marzo) debió ser reemplazado por NaN y luego imputado con ffill (105)
    marzo_idx = df_clean[df_clean['fecha'] == '2023-03-01'].index[0]
    assert df_clean.loc[marzo_idx, 'unidades'] == 105.0
    
    # 3. Validar Relleno de Huecos (Abril)
    # Abril no existía, debió crearse y rellenarse con el valor de Marzo (105)
    abril_row = df_clean[df_clean['fecha'] == '2023-04-01']
    assert not abril_row.empty
    assert abril_row['unidades'].values[0] == 105.0 # ffill de Marzo (que era 0 -> NaN -> 105) o del valor correcto
    
    # Corrijamos la logica de ffill:
    # Enero: 100
    # Febrero: 105
    # Marzo: 0 -> NaN -> 105 (ffill de Feb)
    # Abril (Nuevo): NaN -> 105 (ffill de Mar)
    
    # 4. Validar Detección de Outliers
    # El valor 5000 (Julio) debería ser marcado como atípico
    julio_row = df_clean[df_clean['fecha'] == '2023-07-01']
    assert julio_row['es_atipico'].values[0] == 1
    
    # 5. Validar Tipos de Datos
    assert pd.api.types.is_datetime64_any_dtype(df_clean['fecha'])
    assert pd.api.types.is_float_dtype(df_clean['unidades'])

def test_exclude_current_month(setup_test_environment):
    """Prueba que se corte el mes actual."""
    config_path = setup_test_environment
    
    # Agregar datos del mes actual al raw
    df = pd.read_csv(TEST_CONFIG["paths"]["raw_data"])
    
    today = datetime.now()
    current_month_date = datetime(today.year, today.month, 1).strftime("%Y-%m-%d")
    
    new_row = pd.DataFrame({"fecha": [current_month_date], "unidades_vendidas_mensuales": [200]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TEST_CONFIG["paths"]["raw_data"], index=False)
    
    processor = DataPreprocessor(config_path=config_path)
    df_clean = processor.run_pipeline()
    
    # Verificar que NO exista el mes actual
    assert df_clean[df_clean['fecha'] == current_month_date].empty
