import os
import pandas as pd
import pytest
import yaml
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data_loader import DataLoader

@pytest.fixture
def temp_config(tmp_path):
    """Crea un archivo de configuración temporal para las pruebas."""
    config = {
        "project_name": "Test Project",
        "paths": {
            "raw_data": str(tmp_path / "test_ventas.csv"),
            "reports": str(tmp_path / "reports")
        },
        "data_contract": {
            "columns": {
                "fecha": "datetime",
                "unidades_vendidas_mensuales": "int"
            },
            "allow_extra_columns": False
        },
        "discovery": {
            "sentinel_values_numeric": [0, -1],
            "output_report_name": "test_report.json"
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    return config_file

def create_csv(path, content):
    """Crea un archivo CSV para pruebas."""
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)

def test_load_valid_data(temp_config, tmp_path):
    """Prueba la carga exitosa de datos válidos."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales\n2023-01-01,100\n2023-02-01,200")
    
    loader = DataLoader(config_path=str(temp_config))
    df = loader.load_raw_data()
    
    assert len(df) == 2
    assert df['unidades_vendidas_mensuales'].dtype == 'int32' or df['unidades_vendidas_mensuales'].dtype == 'int64'
    assert pd.api.types.is_datetime64_any_dtype(df['fecha'])

def test_strict_integer_validation_fails(temp_config, tmp_path):
    """Prueba que el cargador falle si hay decimales en una columna entera."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales\n2023-01-01,100.50")
    
    loader = DataLoader(config_path=str(temp_config))
    with pytest.raises(ValueError, match="debe ser ENTERA, pero contiene decimales"):
        loader.load_raw_data()

def test_missing_column_fails(temp_config, tmp_path):
    """Prueba que falle si falta una columna obligatoria."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha\n2023-01-01") # Falta unidades_vendidas_mensuales
    
    loader = DataLoader(config_path=str(temp_config))
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        loader.load_raw_data()

def test_audit_report_content(temp_config, tmp_path):
    """Prueba que el reporte de auditoría contenga las métricas correctas."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales\n2023-01-01,100\n2023-01-01,100\n2023-03-01,0")
    
    loader = DataLoader(config_path=str(temp_config))
    df = loader.load_raw_data()
    report = loader.audit_data(df)
    
    assert report['summary']['total_rows'] == 3
    assert report['summary']['duplicate_rows'] == 1
    assert report['health_checks']['unidades_vendidas_mensuales']['sentinel_values_found'] == 1
    assert 'statistics' in report['health_checks']['unidades_vendidas_mensuales']

def test_extra_columns_fails(temp_config, tmp_path):
    """Prueba que falle si hay columnas extra y el contrato no lo permite."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales,extra_col\n2023-01-01,100,error")
    
    loader = DataLoader(config_path=str(temp_config))
    with pytest.raises(ValueError, match="columnas no permitidas"):
        loader.load_raw_data()

def test_invalid_date_format(temp_config, tmp_path):
    """Prueba que el cargador falle ante fechas con formato inválido."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales\nnot-a-date,100")
    
    loader = DataLoader(config_path=str(temp_config))
    with pytest.raises(Exception):
        loader.load_raw_data()

def test_non_numeric_in_int_column(temp_config, tmp_path):
    """Prueba que falle si hay texto en una columna que debe ser entera."""
    csv_path = tmp_path / "test_ventas.csv"
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales\n2023-01-01,abc")
    
    loader = DataLoader(config_path=str(temp_config))
    with pytest.raises(Exception):
        loader.load_raw_data()

def test_column_name_case_sensitivity(temp_config, tmp_path):
    """Prueba que el cargador falle si los nombres no coinciden exactamente (Case Sensitive)."""
    csv_path = tmp_path / "test_ventas.csv"
    # 'FECHA' en lugar de 'fecha'
    create_csv(csv_path, "FECHA,unidades_vendidas_mensuales\n2023-01-01,100")
    
    loader = DataLoader(config_path=str(temp_config))
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        loader.load_raw_data()

def test_duplicate_dates_different_values(temp_config, tmp_path):
    """Prueba que se detecten fechas duplicadas incluso si los valores de ventas son diferentes."""
    csv_path = tmp_path / "test_ventas.csv"
    # Misma fecha (2023-01-01), diferentes ventas (100 y 150)
    create_csv(csv_path, "fecha,unidades_vendidas_mensuales\n2023-01-01,100\n2023-01-01,150")
    
    loader = DataLoader(config_path=str(temp_config))
    df = loader.load_raw_data()
    report = loader.audit_data(df)
    
    # duplicate_rows sería 0 (porque la fila completa no es igual)
    # duplicate_dates debería ser 1
    assert report['summary']['duplicate_rows'] == 0
    assert report['summary']['duplicate_dates'] == 1

