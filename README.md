# Proyecto: Pronóstico de Ventas - Buñuelos La Floresta

## 📋 Descripción del Proyecto
Este proyecto busca optimizar el sistema de pronóstico de ventas de Buñuelos para la empresa "La Floresta". El objetivo principal es desarrollar un modelo de Machine Learning capaz de predecir las unidades vendidas con un horizonte de 6 meses, reduciendo la dependencia del juicio humano que actualmente presenta desviaciones de hasta el 35%.

## 🎯 Objetivos de Negocio
*   **Reducción de Desviación:** Bajar el error de pronóstico (actualmente en 35%) para minimizar desperdicios de inventario y quiebres de stock.
*   **Alineación Operativa:** Proveer una base técnica fiable para la planeación de compras con proveedores y gestión de insumos.
*   **Mitigación de Sesgos:** Generar un pronóstico objetivo basado en datos, independiente de las presiones comerciales o gerenciales.

## 📏 Reglas de Negocio
*   **Horizonte de Pronóstico:** 6 meses hacia adelante (X+1, X+2, ..., X+6).
*   **Punto de Corte:** En el mes X, se pronostican los meses siguientes. El mes X no se pronostica ya que la información no está cerrada.
*   **Frecuencia:** Datos mensuales.
*   **Ubicación:** Medellín, Colombia (Impacto por Feria de las Flores y festivos locales).

## 📊 Contexto de los Datos
*   **Periodo:** Enero 2018 a Octubre 2025 (8 años de historia).
*   **Hitos Críticos:**
    *   **Pandemia COVID-19:** Caída de ventas entre Mayo 2020 y Febrero 2021.
    *   **Contrato Gran Superficie (Junio 2022):** Incremento estructural en el nivel de ventas.
    *   **Estacionalidad:** Picos en Diciembre/Enero y Junio/Julio.
    *   **Eventos Móviles:** Semana Santa (impacto positivo) y Feria de las Flores (Agosto).
    *   **Factores de Día:** Mayor peso en fines de semana y días festivos.

## 🛠️ Stack Tecnológico sugerido
*   **Librería Core:** `skforecast` (utilizando `ForecasterDirect`).
*   **Algoritmos:**
    *   LightGBM, RandomForest, XGBoost.
    *   GradientBoostingRegressor, HistGradientBoostingRegressor.
    *   Ridge (Modelo lineal de referencia).
*   **Baseline:** Modelo Ingenuo Estacional (Seasonal Naive).

## 📁 Project Structure (Senior Architecture)
This project follows the official senior architecture defined in the `buñuelos-forecaster` skill:

```text
Buñuelos_LaFloresta/
├── data/
│   ├── 01_raw/                 # Immutable source data.
│   ├── 02_cleansed/            # Data after cleaning and handling sentinels.
│   ├── 03_features/            # Intermediate exogenous and calendar datasets.
│   └── 04_processed/           # Final PARQUET dataset ready for modeling.
├── notebooks/                  # EXPERIMENTATION LABORATORY
│   ├── 01_data_discovery.ipynb # Profiling, shapes, duplicates, and null analysis.
│   ├── 02_preprocessing.ipynb  # Cleaning logic and data transformations.
│   ├── 03_eda_business_rules.ipynb # EDA and project-specific rule validation.
│   ├── 04_feature_engineering.ipynb # Exogenous variables (COVID, Holidays, etc.).
│   └── 05_experimentation_backtesting.ipynb # Model training and testing.
├── src/                        # PRODUCTION CODE (Modular logic)
│   ├── data_loader.py          # Loading and ingestion logic.
│   ├── preprocessing.py        # Cleaning and sanitization functions.
│   ├── features.py             # Feature generation (calendar/economic).
│   ├── models.py               # Model training and direct forecasting logic.
│   └── utils.py                # Helper functions (plotting, json reporting).
├── outputs/                    # ASSETS AND ARTEFACTS
│   ├── models/                 # Saved model binaries (.pkl/.joblib).
│   ├── metrics/                # Performance assessment files (CSV/JSON).
│   ├── figures/                # Plots, charts, and visualizations.
│   ├── forecasts/              # Final prediction files.
│   └── reports/                # Step-by-step execution JSON manifests.
├── main.py                     # PIPELINE ORCHESTRATOR
├── .agent/                     # Skills and AI rules.
├── requirements.txt            # Environment dependencies.
└── README.md                   # Project documentation.
```

## 🧠 Filosofía de Trabajo (Trial & Production)
1.  **Laboratorio:** Toda nueva lógica nace en los `notebooks/`.
2.  **Producción:** La lógica validada se refactoriza en módulos `.py` dentro de `src/`.
3.  **Orquestación:** `main.py` ejecuta el pipeline completo y genera reportes de trazabilidad en `outputs/reports/`.


---
**Desarrollado por:** Antigravity AI Assistant
**Ubicación del Proyecto:** `c:\Users\USUARIO\Documents\Forecaster\Buñuelos_LaFloresta`
