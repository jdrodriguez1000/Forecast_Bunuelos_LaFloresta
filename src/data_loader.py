import pandas as pd
import yaml
import json
import os
from datetime import datetime

class DataLoader:
    """
    Componente encargado de la ingesta de datos, validación de esquemas (Data Contract)
    y auditoría de salud de los datos para el proyecto Buñuelos La Floresta.
    """
    
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.paths = self.config['paths']
        self.contract = self.config['data_contract']
        self.discovery_params = self.config['discovery']
        
    def _load_config(self, path):
        """Carga el archivo de configuración YAML."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ No se encontró el archivo de configuración en: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_raw_data(self):
        """
        Carga el archivo raw y aplica el contrato de datos (Schema Enforcement).
        """
        raw_path = self.paths['raw_data']
        print(f"📥 Iniciando carga desde: {raw_path}")
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"❌ El archivo raw no existe en la ruta: {raw_path}")
            
        # Carga inicial
        df = pd.read_csv(raw_path)
        
        # 1. Validación de Nombres y Existencia de Columnas
        expected_cols = list(self.contract['columns'].keys())
        found_cols = list(df.columns)
        
        missing_cols = [c for c in expected_cols if c not in found_cols]
        if missing_cols:
            raise ValueError(f"❌ ERROR DE CONTRATO: Faltan columnas obligatorias: {missing_cols}")
            
        # 2. Validación de Columnas Extra
        if not self.contract.get('allow_extra_columns', True):
            extra_cols = [c for c in found_cols if c not in expected_cols]
            if extra_cols:
                raise ValueError(f"❌ ERROR DE CONTRATO: Se detectaron columnas no permitidas: {extra_cols}")
        
        # 3. Aplicación de Tipos de Datos (Forzado)
        df = self._apply_data_types(df)
        
        print(f"✅ Carga completada. {len(df)} registros validados.")
        return df

    def _apply_data_types(self, df):
        """Asegura que las columnas tengan el tipo definido en el contrato."""
        col_types = self.contract['columns']
        
        for col, dtype in col_types.items():
            if dtype == 'datetime':
                df[col] = pd.to_datetime(df[col])
            elif dtype == 'float':
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            elif dtype == 'int':
                # Validación estricta de enteros: No permitimos decimales no nulos
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                
                # Si hay valores que no son enteros exactos (ej: 295.49), lanzamos error
                if any(numeric_series.dropna() % 1 != 0):
                    bad_values = numeric_series[numeric_series % 1 != 0].head(5).tolist()
                    raise ValueError(f"❌ ERROR DE CONTRATO: La columna '{col}' debe ser ENTERA, pero contiene decimales: {bad_values}")
                
                df[col] = numeric_series.astype(int)
        return df

    def audit_data(self, df):
        """
        Realiza una auditoría médica del DataFrame y genera un reporte de salud.
        Incluye: Nulos, Centinelas, Duplicados, Alta Cardinalidad y Varianza Cero.
        """
        duplicates = int(df.duplicated().sum())
        duplicate_dates = int(df.duplicated(subset=['fecha']).sum())
        
        report = {
            "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "duplicate_rows": duplicates,
                "duplicate_dates": duplicate_dates
            },
            "health_checks": {}
        }
        
        for col in df.columns:
            # 1. Nulos
            null_count = int(df[col].isnull().sum())
            null_pct = (null_count / len(df)) * 100
            
            # 2. Centinelas Numéricos
            sentinel_count = 0
            if pd.api.types.is_numeric_dtype(df[col]):
                sentinels = self.discovery_params.get('sentinel_values_numeric', [])
                sentinel_count = int(df[col].isin(sentinels).sum())
            
            # 3. Cardinalidad
            unique_values = int(df[col].nunique())
            cardinality_ratio = unique_values / len(df)
            
            # 4. Estadísticas Detalladas
            stats = {}
            outliers_info = {}
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = float(df[col].mean())
                std = float(df[col].std()) if len(df) > 1 else 0.0
                q1 = float(df[col].quantile(0.25))
                q3 = float(df[col].quantile(0.75))
                iqr = q3 - q1
                
                # IQR Outliers
                lower_iqr = q1 - 1.5 * iqr
                upper_iqr = q3 + 1.5 * iqr
                outliers_iqr = df[(df[col] < lower_iqr) | (df[col] > upper_iqr)]
                
                # Z-Score Outliers
                import numpy as np
                from scipy import stats as scipy_stats
                z_scores = np.abs(scipy_stats.zscore(df[col]))
                outliers_z = df[z_scores > 3]

                outliers_info = {
                    "method_iqr": {
                        "lower_bound": round(lower_iqr, 2),
                        "upper_bound": round(upper_iqr, 2),
                        "count": int(len(outliers_iqr)),
                        "indices": outliers_iqr.index.tolist()
                    },
                    "method_zscore": {
                        "count": int(len(outliers_z)),
                        "indices": outliers_z.index.tolist()
                    }
                }

                stats = {
                    "mean": round(mean, 2),
                    "median": float(df[col].median()),
                    "variance": float(df[col].var()) if len(df) > 1 else 0.0,
                    "std": round(std, 2),
                    "kurtosis": float(df[col].kurt()) if len(df) > 1 else 0.0,
                    "skewness": float(df[col].skew()) if len(df) > 1 else 0.0,
                    "p25": q1,
                    "p75": q3,
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
                is_constant = unique_values <= 1
            else:
                # Estadísticas para variables no numéricas (Categorical/Datetime)
                stats = {
                    "mode": str(df[col].mode().iloc[0]) if not df[col].empty else None,
                    "min": str(df[col].min()) if not df[col].empty else None,
                    "max": str(df[col].max()) if not df[col].empty else None
                }
                is_constant = unique_values <= 1
            
            report["health_checks"][col] = {
                "null_count": null_count,
                "null_percentage": round(null_pct, 2),
                "sentinel_values_found": sentinel_count,
                "unique_values": unique_values,
                "cardinality_ratio": round(cardinality_ratio, 4),
                "is_constant": is_constant,
                "data_type": str(df[col].dtype),
                "statistics": stats,
                "outliers": outliers_info
            }
            
        return report

    def save_report(self, report, phase_name="discovery"):
        """Guarda el reporte de auditoría en formato JSON."""
        report_dir = self.paths['reports']
        os.makedirs(report_dir, exist_ok=True)
        
        file_name = self.discovery_params.get('output_report_name', f"phase_{phase_name}_report.json")
        save_path = os.path.join(report_dir, file_name)
        
        # Clase para codificar tipos especiales en JSON
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                return super(DateTimeEncoder, self).default(obj)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False, cls=DateTimeEncoder)
            
        print(f"📄 Reporte de auditoría guardado en: {save_path}")

    def generate_outlier_plot(self, df, column='unidades_vendidas_mensuales'):
        """Genera y guarda el gráfico de outliers."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig_dir = self.paths['figures']
        os.makedirs(fig_dir, exist_ok=True)
        
        # Calcular outliers para el gráfico
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        
        plt.figure(figsize=(15, 6))
        
        # Subplot 1: Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df[column], color='#AED6F1')
        plt.title(f'Boxplot de {column}')
        
        # Subplot 2: Serie con Outliers
        plt.subplot(1, 2, 2)
        plt.plot(df['fecha'], df[column], color='#2E86C1', alpha=0.5, label='Ventas')
        plt.scatter(outliers['fecha'], outliers[column], color='red', marker='x', label='Outliers (IQR)')
        plt.title('Outliers en la Serie Temporal')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(fig_dir, f'outliers_{column}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️ Gráfico de outliers guardado en: {save_path}")

if __name__ == "__main__":
    import sys
    # Asegurar que la consola soporte caracteres UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    # Prueba rápida del componente
    try:
        loader = DataLoader()
        data = loader.load_raw_data()
        health_report = loader.audit_data(data)
        loader.save_report(health_report)
    except Exception as e:
        print(f"⚠️ Error en la ejecución del DataLoader: {e}")
