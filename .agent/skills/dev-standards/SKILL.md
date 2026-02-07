---
name: dev-standards
description: Estándares de ingeniería de software, arquitectura de proyectos y protocolo de control de versiones (Git) para desarrollo profesional.
---

# Skill: Estándares de Desarrollo Profesional (Dev-Standards)

Esta habilidad define la infraestructura, organización y protocolos de comunicación técnica del proyecto. Asegura que el código sea mantenible, escalable y profesional.

## 📂 Arquitectura del Proyecto (Estándar Inglés)

El proyecto debe seguir estrictamente esta jerarquía de directorios:

```text
Project_Root/
├── data/
│   ├── 01_raw/                 # Datos inmutables de origen.
│   ├── 02_cleansed/            # Datos tras limpieza inicial.
│   ├── 03_features/            # Dataset de variables intermedias.
│   └── 04_processed/           # Dataset final (formato PARQUET).
├── notebooks/                  # LABORATORIO DE EXPERIMENTACIÓN
│   ├── 01_data_discovery.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda_business_rules.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_experimentation_backtesting.ipynb
├── src/                        # CÓDIGO PRODUCTIVO (Lógica modular)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
├── outputs/                    # ARTEFACTOS Y RESULTADOS
│   ├── models/                 # Binarios de modelos entrenados.
│   ├── metrics/                # Archivos de desempeño estadístico.
│   ├── figures/                # Gráficas y visualizaciones.
│   ├── forecasts/              # Salidas de predicción final.
│   └── reports/                # Reportes JSON de trazabilidad paso a paso.
├── main.py                     # ORQUESTADOR CENTRAL
├── requirements.txt            # Dependencias.
└── README.md                   # Documentación.
```

## 🚀 Protocolo de Git y GitHub (Conventional Commits)

Cada commit debe seguir la convención de mensajes estructurados:

- `feat:` Nuevas funcionalidades, carpetas o archivos base.
- `fix:` Corrección de bugs o errores en el código/datos.
- `docs:` Cambios en README, habilidades o documentación técnica.
- `refactor:` Mejoras en el código que no cambian el comportamiento.
- `chore:` Tareas de mantenimiento (actualizar `.gitignore`, dependencias).
- `test:` Adición o corrección de pruebas.

**Ejemplo:** `feat: create initial project structure and data directories`

## 🧠 Filosofía de Ingeniería

1.  **Laboratorio vs Producción:**
    - Se explora y valida la lógica exclusivamente en los `notebooks/`.
    - La lógica exitosa se refactoriza en módulos `.py` dentro de `src/`.
    - `main.py` orquesta la ejecución final invocando los módulos de `src/`.
2.  **Trazabilidad JSON:**
    - Cada fase significativa del proceso debe generar un archivo `.json` en `outputs/reports/`.
    - El reporte debe capturar metadatos, contadores de registros y estados de validación.
3.  **Naming & Language:**
    - Todo el proyecto (carpetas, archivos, variables) debe estar en **INGLÉS**.
    - Se privilegia el uso de `snake_case`.

## 📋 Checklist de Calidad
- [ ] ¿La estructura de carpetas es idéntica al diagrama?
- [ ] ¿El mensaje de Git sigue el estándar Conventional Commits?
- [ ] ¿Toda la nomenclatura está en inglés?
- [ ] ¿La lógica probada en el notebook ya está en `src/`?
- [ ] ¿Se generó el reporte JSON de trazabilidad?
