---
trigger: always_on
---

# Reglas del Proyecto: Pronóstico Buñuelos La Floresta

Este archivo contiene las instrucciones críticas que deben seguir todos los agentes de IA al trabajar en este repositorio.

## 🧠 Lógica de Negocio y Pronóstico
- **Horizonte de Tiempo:** Los pronósticos DEBEN ser siempre a 6 meses (X+1 a X+6).
- **Desfase de Información:** Nunca usar datos del mes actual (X) para predecir el futuro inmediato, ya que la información del mes de ejecución se considera incompleta.
- **Validación:** Se debe utilizar `backtesting` de series temporales (específicamente de la librería `skforecast`) para validar los modelos.

## 🛠️ Stack Tecnológico y Modelado
- **Librería Primaria:** `skforecast`.
- **Estrategia de Modelado:** Se DEBE utilizar `ForecasterDirect`. No utilizar modelos recursivos simples sin una justificación de negocio sólida.
- **Modelos Obligatorios:** En cada experimento se deben comparar los siguientes modelos:
  - `Ridge` (Baseline lineal)
  - `RandomForestRegressor`
  - `LGBMRegressor`
  - `XGBRegressor`
  - `GradientBoostingRegressor`
  - `HistGradientBoostingRegressor`
- **Línea Base (Baseline):** Antes de cualquier modelo de ML, se debe implementar un modelo "Ingenuo Estacional" (Seasonal Naive) como punto de referencia.

## 📊 Ingeniería de Características (Exógenas)
Los modelos DEBEN incluir variables exógenas para capturar la realidad del negocio:
1.  **Hitos Estructurales:**
    - Variable binaria para el periodo COVID: Mayo 2020 - Febrero 2021.
    - Variable binaria para el contrato de Gran Superficie: Junio 2022 en adelante.
2.  **Efecto Calendario:**
    - Contador de días festivos por mes (Colombia).
    - Contador de fines de semana (Sábados/Domingos) por mes.
    - Indicador de Semana Santa (Móvil).
    - Indicador de Feria de las Flores (Agosto - Medellín).
3.  **Variables Económicas:** Incorporar Inflación (IPC) y TRM como variables de prueba.

## 📁 Estructura y Código
- Seguir la estructura de carpetas: `data/01_raw`, `data/02_cleansed`, `notebooks/`, `src/`.
- Documentar en los notebooks el "Por qué" de cada variable creada, haciendo referencia a este archivo de reglas.
- Mantener una semilla de aleatoriedad (`random_state`) fija para reproducibilidad.
