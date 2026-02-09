import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.features import (
    add_calendar_features,
    add_structural_flags,
    integrate_macro_data,
    run_feature_engineering_pipeline
)

# --- Fixtures ---

@pytest.fixture
def sample_config():
    return {
        'paths': {
            'cleansed_data': 'test_data',
            'features_data': 'test_data',
            'processed_data': 'test_outputs',
            'figures': 'test_outputs',
            'reports': 'test_outputs'
        },
        'feature_engineering': {
            'input': {
                'sales_file': 'sales_clean.csv',
                'macro_file': 'macro_test.csv'
            },
            'output': {
                'filename': 'test_features.parquet',
                'report_name': 'test_fe_report.json'
            },
            'calendar': {
                'country': 'Colombia',
                'features': ['festivos_conteo', 'fines_semana_conteo', 'es_semana_santa', 'es_feria_flores']
            },
            'structural_flags': {
                'covid_impact': {'start': '2020-05-01', 'end': '2021-02-28'},
                'retail_expansion': {'start': '2022-06-01'}
            },
            'macro_selection': {
                'columns': ['ipc_mensual', 'trm_promedio']
            }
        }
    }

@pytest.fixture
def sample_sales_df():
    # Creamos un dataset de 3 meses significativos
    dates = pd.to_datetime(['2020-04-01', '2020-05-01', '2022-07-01'])
    df = pd.DataFrame({
        'unidades': [1000, 800, 5000],
        'es_atipico': [0, 1, 0]
    }, index=dates)
    df.index.name = 'fecha'
    return df

@pytest.fixture
def sample_macro_df():
    dates = pd.to_datetime(['2020-04-01', '2020-05-01', '2022-07-01'])
    df = pd.DataFrame({
        'ipc_mensual': [0.5, 0.4, 0.8],
        'trm_promedio': [3800.0, 3900.0, 4200.0], # Forzamos float
        'tasa_desempleo': [10.0, 12.0, 9.0]
    }, index=dates)
    df.index.name = 'fecha'
    return df

# --- Tests ---

def test_add_calendar_features(sample_sales_df):
    """Verifica la creación de variables de calendario."""
    df_result = add_calendar_features(sample_sales_df)
    
    # Verificar columnas creadas
    expected_cols = ['festivos_conteo', 'fines_semana_conteo', 'es_semana_santa', 'es_feria_flores']
    for col in expected_cols:
        assert col in df_result.columns
        
    # Verificar que no hay nulos en calendarios
    assert df_result[expected_cols].isna().sum().sum() == 0
    
    # 2020-04-01 fue Semana Santa (Viernes Santo fue el 10 de Abril)
    assert df_result.loc['2020-04-01', 'es_semana_santa'] == 1
    
    # 2022-07-01 NO es Feria de Flores (es en Agosto)
    assert df_result.loc['2022-07-01', 'es_feria_flores'] == 0

def test_add_structural_flags(sample_sales_df, sample_config):
    """Verifica la asignación de flags COVID y Retail."""
    df_result = add_structural_flags(sample_sales_df, sample_config)
    
    # 2020-04-01: Antes del periodo COVID marcado en config (Mayo 2020)
    assert df_result.loc['2020-04-01', 'flag_covid'] == 0
    
    # 2020-05-01: Inicio COVID
    assert df_result.loc['2020-05-01', 'flag_covid'] == 1
    
    # 2022-07-01: Después del inicio de Retail Expansion (Junio 2022)
    assert df_result.loc['2022-07-01', 'flag_retail'] == 1
    assert df_result.loc['2020-04-01', 'flag_retail'] == 0

def test_integrate_macro_data(sample_sales_df, sample_macro_df, sample_config):
    """Verifica que se unan solo las columnas macro seleccionadas y se manejen nulos."""
    df_result = integrate_macro_data(sample_sales_df, sample_macro_df, sample_config)
    
    # Verificar columnas seleccionadas
    assert 'ipc_mensual' in df_result.columns
    assert 'trm_promedio' in df_result.columns
    assert 'tasa_desempleo' not in df_result.columns # No estaba en macro_selection
    
    # Verificar que no hay nulos (ffill/bfill)
    assert df_result.isna().sum().sum() == 0

def test_feature_data_types(sample_sales_df, sample_macro_df, sample_config):
    """Verifica que los tipos de datos sean los correctos para cada tipo de variable."""
    # Ejecutamos las transformaciones
    df = add_calendar_features(sample_sales_df)
    df = add_structural_flags(df, sample_config)
    df = integrate_macro_data(df, sample_macro_df, sample_config)
    
    # Definir tipos esperados
    expected_types = {
        'unidades': ['int64', 'float64'], # Depende de carga inicial pero debe ser numérico
        'es_atipico': ['int64', 'int32'],
        'festivos_conteo': ['int64', 'int32'],
        'fines_semana_conteo': ['int64', 'int32'],
        'es_semana_santa': ['int64', 'int32'],
        'es_feria_flores': ['int64', 'int32'],
        'flag_covid': ['int64', 'int32'],
        'flag_retail': ['int64', 'int32'],
        'ipc_mensual': ['float64'],
        'trm_promedio': ['float64']
    }
    
    for col, types in expected_types.items():
        assert col in df.columns, f"Columna {col} no encontrada"
        actual_type = str(df[col].dtype)
        # Verificamos que el tipo actual coincida con alguno de los permitidos
        assert any(t in actual_type for t in types), f"Columna {col} tiene tipo {actual_type}, esperaba uno de {types}"

@patch('src.features.load_config')
@patch('src.features.load_input_data')
@patch('src.features.save_feature_visualizations')
@patch('pandas.DataFrame.to_parquet')
@patch('os.makedirs')
@patch('json.dump')
@patch('builtins.open')
def test_feature_pipeline_orchestration(
    mock_open, mock_json, mock_makedirs, mock_parquet, mock_viz,
    mock_load_data, mock_load_config, sample_config, sample_sales_df, sample_macro_df
):
    """Prueba que el pipeline orquestador ejecute todos los pasos y genere el reporte JSON."""
    mock_load_config.return_value = sample_config
    mock_load_data.return_value = (sample_sales_df, sample_macro_df)
    
    run_feature_engineering_pipeline('dummy_path.yaml')
    
    # Verificar que se intentó guardar el parquet
    assert mock_parquet.called
    
    # Verificar que se llamó a la visualización
    assert mock_viz.called
    
    # Verificar que se generó un reporte JSON
    assert mock_json.called
    args, _ = mock_json.call_args
    report_content = args[0]
    
    assert report_content['phase'] == '04_Feature_Engineering'
    assert 'data_quality_validation' in report_content
    assert report_content['data_quality_validation']['total_nulls_detected'] == 0
    assert 'output_metrics' in report_content
    assert 'schema_description' in report_content['output_metrics']
    # unidades + es_atipico + 4 cal + 2 flags + 2 macro = 10
    assert len(report_content['output_metrics']['schema_description']) == 10 

