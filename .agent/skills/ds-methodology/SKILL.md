---
name: ds-methodology
description: Metodología estándar para auditoría, descubrimiento y análisis de datos. Define los requisitos de calidad y visualización para cada fase analítica.
---

# Skill: Metodología de Ciencia de Datos (DS-Methodology)

Esta habilidad establece los estándares analíticos y de auditoría que deben aplicarse en todo proyecto de datos, garantizando hallazgos profundos y visualizaciones accionables.

## 🔬 Fase 1: Data Discovery (Auditoría Médica)

En la fase inicial de carga, el agente debe realizar obligatoriamente los siguientes diagnósticos:

### 1. Auditoría de Salud (Nulos vs. Centinelas)
- **Missing Values:** Identificar `NaN` o celdas vacías.
- **Sentinel Values Search:** Buscar valores ficticios por tipo de dato:
    - **Numéricos:** `0`, `-1`, `99`, `999`.
    - **Categorícos:** `"N/A"`, `"Unknown"`, `"Empty"`, `" "`.
    - **Booleanos:** Valores fuera de `True/False` (ej. `2`, `-1`).
    - **Datetime:** Fechas extremas (`1900-01-01`, `2099-12-31`).

### 2. Capacidad Informativa (Varianza y Cardinalidad)
- **Zero Variance:** Identificar columnas constantes (no aportan información).
- **High Cardinality:** Detectar variables tipo ID o de varianza extrema que puedan causar sobreajuste.

### 3. Perfilado Estadístico y de Pesos
- **Estadísticas Descriptivas:** Media, mediana, desviación estándar y percentiles para numéricos.
- **Análisis de Pesos Categorícos:** Para cada categoría, informar:
    - Lista de valores únicos.
    - Frecuencia absoluta (conteo).
    - Frecuencia relativa (% de peso sobre el total).

## 📊 Estándares de Visualización

Toda fase analítica debe estar acompañada de gráficas que faciliten la interpretación del negocio:

### 📈 Visualización del Target (Ventas/Variable Objetivo)
- **Time Series Plot:** Línea de tiempo para identificar tendencia y estacionalidad.
- **Seasonal Plot:** Gráfica por meses o años para confirmar patrones cíclicos.

### 🔍 Visualización de Calidad y Distribución
- **Matrix de Nulos/Centinelas:** Representación visual de dónde faltan datos (ej. Mapa de calor).
- **Histogramas / Boxplots:** Para entender la dispersión y detectar outliers de forma visual.
- **Bar Charts de Pesos:** Para variables categóricas, mostrando el TOP de categorías y su dominancia.

## 🧪 Fase de Experimentación (Rigor Senior)

1.  **Baseline Obligatorio:** Siempre comparar el modelo sugerido contra un modelo ingenuo (Naive) o estacional simple.
2.  **Backtesting:** Uso de validación cruzada temporal para medir la robustez del modelo.
3.  **Importancia de Variables:** Graficar siempre qué variables (exógenas o lags) están impactando más la predicción.

## 📋 Lista de Verificación (Metodología)
- [ ] ¿Se buscaron centinelas en tipos: numérico, texto, fecha y booleano?
- [ ] ¿Se analizó la frecuencia y peso porcentual de las categorías?
- [ ] ¿Se identificaron variables de varianza cero o IDs innecesarios?
- [ ] ¿Hay visualizaciones de tendencia y estacionalidad?
- [ ] ¿Se incluyó una matriz o gráfica de salud de datos (nulos/centinelas)?
