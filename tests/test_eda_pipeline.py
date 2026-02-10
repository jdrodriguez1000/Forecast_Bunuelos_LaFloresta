import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.eda_pipeline import (
    perform_splitting, 
    analyze_drift, 
    run_stationarity_test, 
    analyze_calendar_effects
)

# --- Fixtures ---

@pytest.fixture
def sample_config():
    return {
        'paths': {
            'cleansed_data': 'test_data',
            'features_data': 'test_data',
            'figures': 'test_outputs',
            'reports': 'test_outputs',
            'metrics': 'test_outputs'
        },
        'eda': {
            'input': {
                'sales_file': 'sales.csv',
                'macro_file': 'macro.csv'
            },
            'output': {
                'figures_dir': 'eda',
                'report_name': 'test_report.json'
            },
            'splitting': {
                'train': {'start_date': '2023-01-01', 'end_date': '2023-03-01'},
                'validation': {'start_date': '2023-04-01', 'end_date': '2023-05-01'},
                'test': {'start_date': '2023-06-01', 'end_date': '2023-07-01'}
            },
            'business_rules': {
                'structural_breaks': {
                    'covid_period': {'start': '2020-03-01', 'end': '2021-02-01'},
                    'gran_superficie_contract': {'start': '2022-06-01'}
                }
            }
        }
    }

@pytest.fixture
def sample_df():
    dates = pd.date_range(start='2023-01-01', end='2023-07-01', freq='MS')
    df = pd.DataFrame({
        'unidades': [100, 110, 105, 200, 210, 300, 310]
    }, index=dates)
    df.index.name = 'fecha'
    return df

# --- Tests ---

def test_perform_splitting(sample_df, sample_config):
    """Verifica que el splitting respete las fechas del config."""
    train, val, test = perform_splitting(sample_df, sample_config)
    
    assert len(train) == 3
    assert len(val) == 2
    assert len(test) == 2
    
    assert train.index.max() == pd.Timestamp('2023-03-01')
    assert val.index.min() == pd.Timestamp('2023-04-01')
    assert test.index.min() == pd.Timestamp('2023-06-01')

def test_perform_splitting_dynamic(sample_df):
    """Verifica la lógica de división dinámica basada en meses."""
    dynamic_config = {
        'eda': {
            'splitting': {
                'method': 'dynamic',
                'test_months': 2,
                'validation_months': 2
            }
        }
    }
    # sample_df tiene 7 meses (Enero a Julio 2023)
    train, val, test = perform_splitting(sample_df, dynamic_config)
    
    assert len(test) == 2      # Junio, Julio
    assert len(val) == 2       # Abril, Mayo
    assert len(train) == 3     # Enero, Febrero, Marzo
    
    assert test.index.max() == pd.Timestamp('2023-07-01')
    assert val.index.max() == pd.Timestamp('2023-05-01')
    assert train.index.max() == pd.Timestamp('2023-03-01')


def test_analyze_drift(sample_df, sample_config):
    """Verifica el cálculo de drift."""
    train, val, test = perform_splitting(sample_df, sample_config)
    
    # Debug info
    print(f"Train size: {len(train)}")
    print(f"Train mean: {train['unidades'].mean()}")
    print(f"Val mean: {val['unidades'].mean()}")
    
    stats = analyze_drift(train, val, test)
    
    # Train mean: (100+110+105)/3 = 105
    # Val mean: (200+210)/2 = 205
    # Drift % = ((205 - 105) / 105) * 100 = 95.23%
    
    assert stats['train_mean'] == 105.0, f"Train Mean: {stats.get('train_mean')}"
    assert stats['val_mean'] == 205.0, f"Val Mean: {stats.get('val_mean')}"
    assert stats['drift_detected'], f"Drift Detected: {stats.get('drift_detected')} (Pct: {stats.get('drift_val_vs_train_pct')})"
    assert 95.0 < stats['drift_val_vs_train_pct'] < 96.0, f"Drift Pct: {stats.get('drift_val_vs_train_pct')}"

@patch('src.eda_pipeline.adfuller')
def test_run_stationarity_test(mock_adfuller):
    """Verifica que el resultado del test ADF se formatee correctamente."""
    # Mock return: (adf_stat, p_value, usedlag, nobs, critical_values, icbest)
    mock_adfuller.return_value = (-2.5, 0.15, 0, 10, {}, 100)
    
    series = pd.Series([1, 2, 3, 4, 5])
    result = run_stationarity_test(series)
    
    assert result['adf_statistic'] == -2.5
    assert result['p_value'] == 0.15
    assert result['is_stationary'] is False

@patch('src.eda_pipeline.plt')
@patch('src.eda_pipeline.sns')
def test_analyze_calendar_effects(mock_sns, mock_plt, sample_df, tmp_path):
    """Verifica la creación de features de calendario y que no falle la graficación."""
    
    # Mock plt.subplots to return a tuple (fig, ax)
    mock_axes = MagicMock()
    # Configure axes to be accessible by index like axes[0, 0]
    mock_axes.__getitem__.return_value.__getitem__.return_value = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)

    # Usar tmp_path fixture de pytest para directorio temporal
    output_dir = str(tmp_path)
    
    effects = analyze_calendar_effects(sample_df, output_dir)
    
    # La función debe retornar un dict con promedios
    assert 'avg_sales_feria' in effects
    assert 'avg_sales_normal' in effects
    
    # Verifica que se intentó guardar la figura
    mock_plt.savefig.assert_called()

def test_numpy_conversion_logic():
    """Replica y valida la lógica de conversión JSON para tipos NumPy."""
    # Como la función es interna, replicamos la lógica para asegurar que el patrón es correcto
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return str(obj)

    assert convert_numpy(np.int64(10)) == 10.0
    assert convert_numpy(np.float64(10.5)) == 10.5
    assert convert_numpy(np.array([1, 2])) == [1, 2]
    assert convert_numpy(np.bool_(True)) == True
    assert convert_numpy(np.nan) is None

@patch('src.eda_pipeline.seasonal_decompose')
@patch('src.eda_pipeline.plot_acf')
@patch('src.eda_pipeline.plot_pacf')
@patch('src.eda_pipeline.run_stationarity_test')
@patch('src.eda_pipeline.analyze_calendar_effects')
@patch('src.eda_pipeline.load_data')
@patch('src.eda_pipeline.load_config')
@patch('src.eda_pipeline.plt')
@patch('src.eda_pipeline.sns')
@patch('src.eda_pipeline.json.dump')
@patch('builtins.open')
@patch('os.listdir')
@patch('os.makedirs')
def test_data_leakage_prevention_in_analysis(
    mock_makedirs, mock_listdir, mock_open, mock_json_dump, mock_sns, mock_plt,
    mock_load_config, mock_load_data, 
    mock_calendar, mock_stationarity, mock_plot_pacf, mock_plot_acf, mock_decompose,
    sample_config, sample_df
):
    """
    Test de Seguridad Crítico:
    Verifica que los análisis de EDA (Descomposición, ADF, ACF)
    se ejecuten ESTRICTAMENTE sobre el conjunto de TRAIN.
    """
    from src.eda_pipeline import run_eda_analysis
    
    # 1. Configurar Mocks para devolver datos de prueba
    mock_load_config.return_value = sample_config
    mock_load_data.return_value = (sample_df, None)
    
    # Evitar errores de atributo en mocks anidados
    mock_decompose.return_value.plot.return_value.set_size_inches = MagicMock()
    mock_stationarity.return_value = {'p_value': 0.05, 'is_stationary': False, 'adf_statistic': -1.0}
    mock_calendar.return_value = {}
    mock_listdir.return_value = ['dummy.png']
    
    # Configure plt.subplots to return a tuple (fig, axes) to avoid unpacking error
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    
    # 2. Ejecutar el Pipeline (Orquestador)
    # Pasamos una ruta dummy porque load_config está mockeado
    run_eda_analysis('dummy_config.yaml')
    
    # 3. Validaciones de Fuga de Datos (Data Leakage)
    
    # Fecha de corte de Train según fixture: 2023-03-01
    # Datos completos van hasta: 2023-07-01
    train_end_date = pd.Timestamp(sample_config['eda']['splitting']['train']['end_date'])
    
    # A. Validar Descomposición Estacional
    assert mock_decompose.called, "Se debe ejecutar seasonal_decompose"
    args, _ = mock_decompose.call_args
    # args[0] es la serie pasada a la función
    max_date_decompose = args[0].index.max()
    assert max_date_decompose <= train_end_date, \
        f"🚨 DATA LEAKAGE: Descomposición usó datos hasta {max_date_decompose} (Train corta en {train_end_date})"

    # B. Validar Test de Estacionariedad
    assert mock_stationarity.called
    args, _ = mock_stationarity.call_args
    max_date_adf = args[0].index.max()
    assert max_date_adf <= train_end_date, \
        f"🚨 DATA LEAKAGE: Test ADF usó datos hasta {max_date_adf}"

    # C. Validar Autocorrelación
    assert mock_plot_acf.called
    args, _ = mock_plot_acf.call_args
    max_date_acf = args[0].index.max()
    assert max_date_acf <= train_end_date, \
        f"🚨 DATA LEAKAGE: Plot ACF usó datos hasta {max_date_acf}"
