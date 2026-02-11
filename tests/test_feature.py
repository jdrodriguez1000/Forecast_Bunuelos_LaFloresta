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
    add_cyclic_features,
    add_structural_flags,
    add_intensity_features,
    add_momentum_features,
    impute_momentum_nulls,
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
                'features': ['festivos_conteo', 'fines_semana_conteo', 'es_semana_santa', 'es_feria_flores', 'mes_sin', 'mes_cos', 'trimestre_sin', 'trimestre_cos']
            },
            'structural_flags': {
                'covid_impact': {'start': '2020-05-01', 'end': '2021-02-28'},
                'retail_expansion': {'start': '2022-06-01'},
                'contract_maturation': {'reference_date': '2022-06-01'}
            },
            'intensity_mapping': {
                'super_pico': [12],
                'pico_alto': [6, 8],
                'pico_medio': [5, 11]
            },
            'growth_momentum': {
                'yoy_comparison': True,
                'rolling_windows': [3, 6]
            },
            'macro_selection': {
                'columns': ['ipc_mensual', 'trm_promedio', 'ipc_alimentos_mensual']
            }
        }
    }

@pytest.fixture
def sample_sales_df():
    # Creamos un dataset de 14 meses para probar YoY y medias móviles
    dates = pd.date_range(start='2020-01-01', periods=14, freq='MS')
    df = pd.DataFrame({
        'unidades': [1000 + i*100 for i in range(len(dates))], # Crecimiento lineal para tests
        'es_atipico': [0] * len(dates)
    }, index=dates)
    df.index.name = 'fecha'
    # Forzar algunos casos específicos
    # Abril 2020 fue Semana Santa
    return df

@pytest.fixture
def sample_macro_df():
    dates = pd.date_range(start='2020-01-01', periods=14, freq='MS')
    df = pd.DataFrame({
        'ipc_mensual': [0.5] * len(dates),
        'trm_promedio': [3800.0] * len(dates),
        'tasa_desempleo': [10.0] * len(dates)
        # Omitimos ipc_alimentos_mensual para probar el proxy en integrate_macro_data
    }, index=dates)
    df.index.name = 'fecha'
    return df

# --- Tests ---

def test_add_calendar_features(sample_sales_df):
    """Verifica la creación de variables de calendario extendidas."""
    df_result = add_calendar_features(sample_sales_df)
    
    # Verificar columnas creadas
    expected_cols = [
        'festivos_conteo', 'fines_semana_conteo', 'es_semana_santa', 
        'conteo_novenas', 'es_puente_festivo', 'efecto_quincena'
    ]
    for col in expected_cols:
        assert col in df_result.columns
        
    # Verificar que no hay nulos
    assert df_result[expected_cols].isna().sum().sum() == 0
    
    # Diciembre debe tener novenas (conteo > 0)
    dec_date = pd.to_datetime('2020-12-01')
    if dec_date in df_result.index:
        assert df_result.loc[dec_date, 'conteo_novenas'] == 9
    
    # Abril 2020 fue Semana Santa
    assert df_result.loc['2020-04-01', 'es_semana_santa'] == 1

def test_add_cyclic_features(sample_sales_df):
    """Verifica que se generen correctamente las variables cíclicas."""
    df_result = add_cyclic_features(sample_sales_df)
    
    cyclic_cols = ['mes_sin', 'mes_cos', 'trimestre_sin', 'trimestre_cos']
    for col in cyclic_cols:
        assert col in df_result.columns
        assert df_result[col].min() >= -1.05 # Tolerancia por float
        assert df_result[col].max() <= 1.05
    
    # Abril (mes 4): sin(2*pi*4/12) = 0.866
    assert np.isclose(df_result.loc['2020-04-01', 'mes_sin'], 0.866025, atol=1e-5)

def test_add_structural_flags(sample_sales_df, sample_config):
    """Verifica la asignación de flags y maduración."""
    df_result = add_structural_flags(sample_sales_df, sample_config)
    
    assert 'flag_covid' in df_result.columns
    assert 'flag_retail' in df_result.columns
    assert 'maduracion_contrato' in df_result.columns
    
    # Retail inicia en 2022-06-01. En 2020 es 0.
    assert df_result.loc['2020-05-01', 'flag_retail'] == 0
    
    # Maduración debe ser >= 0
    assert (df_result['maduracion_contrato'] >= 0).all()

def test_add_intensity_features(sample_sales_df, sample_config):
    """Verifica el mapeo de intensidad estacional."""
    df_result = add_intensity_features(sample_sales_df, sample_config)
    
    assert 'intensidad_comercial' in df_result.columns
    # Diciembre debe ser 3 (Super Pico)
    assert df_result.loc['2020-12-01', 'intensidad_comercial'] == 3
    # Enero debe ser 0 (Normal)
    assert df_result.loc['2020-01-01', 'intensidad_comercial'] == 0

def test_add_momentum_features(sample_sales_df, sample_config):
    """Verifica cálculo de crecimiento YoY y momentum."""
    df_result = add_momentum_features(sample_sales_df, sample_config)
    
    assert 'crecimiento_yoy' in df_result.columns
    assert 'growth_momentum_3m' in df_result.columns
    
    # El mes 13 (2021-01-01) debería tener crecimiento YoY vs 2020-01-01
    yoy_val = df_result.loc['2021-01-01', 'crecimiento_yoy']
    # Unidades 2020-01: 1000, 2021-01 (mes 13): 1000 + 12*100 = 2200
    # expected = (2200 - 1000) / 1000 = 1.2
    assert np.isclose(yoy_val, 1.2)

def test_impute_momentum_nulls(sample_sales_df, sample_config):
    """Verifica que el backfill elimine los nulos iniciales de momentum."""
    df = add_momentum_features(sample_sales_df, sample_config)
    # Al inicio hay nulos (primeros 12 meses para YoY)
    assert df['crecimiento_yoy'].isna().any()
    
    df_fixed = impute_momentum_nulls(df)
    assert not df_fixed['crecimiento_yoy'].isna().any()
    # El valor de 2020-01-01 debe ser igual al primer valor no nulo (2021-01-01)
    assert df_fixed.loc['2020-01-01', 'crecimiento_yoy'] == df_fixed.loc['2021-01-01', 'crecimiento_yoy']

def test_integrate_macro_data(sample_sales_df, sample_macro_df, sample_config):
    """Verifica la unión macro y el uso de Proxys."""
    # Omitimos 'ipc_alimentos_mensual' en sample_macro_df para que use el proxy de 'ipc_mensual'
    df_result = integrate_macro_data(sample_sales_df, sample_macro_df, sample_config)
    
    assert 'ipc_mensual' in df_result.columns
    assert 'ipc_alimentos_mensual' in df_result.columns
    # Deben ser iguales por el proxy
    assert (df_result['ipc_alimentos_mensual'] == df_result['ipc_mensual']).all()

def test_feature_data_types(sample_sales_df, sample_macro_df, sample_config):
    """Verifica los tipos de datos del dataset final consolidado."""
    df = add_calendar_features(sample_sales_df)
    df = add_cyclic_features(df)
    df = add_structural_flags(df, sample_config)
    df = add_intensity_features(df, sample_config)
    df = add_momentum_features(df, sample_config)
    df = integrate_macro_data(df, sample_macro_df, sample_config)
    df = impute_momentum_nulls(df)
    
    expected_cols = [
        'unidades', 'conteo_novenas', 'es_puente_festivo', 'maduracion_contrato',
        'intensidad_comercial', 'crecimiento_yoy', 'growth_momentum_3m', 
        'ipc_alimentos_mensual'
    ]
    for col in expected_cols:
        assert col in df.columns, f"Falta la columna {col}"

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
    """Prueba que el pipeline orquestador ejecute todos los pasos y genere el reporte detallado."""
    mock_load_config.return_value = sample_config
    mock_load_data.return_value = (sample_sales_df, sample_macro_df)
    
    run_feature_engineering_pipeline('dummy_path.yaml')
    
    assert mock_parquet.called
    assert mock_json.called
    
    args, _ = mock_json.call_args
    report = args[0]
    
    assert report['phase'] == '04_Feature_Engineering'
    assert 'column_details' in report
    assert 'features_created' in report
    # Verificar que el reporte incluye las nuevas categorías
    assert 'intensity' in report['features_created']
    assert 'momentum' in report['features_created']
 

