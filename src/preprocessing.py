import pandas as pd
import numpy as np
import yaml
import os
import json
from datetime import datetime

class DataPreprocessor:
    """
    Clase encargada de la limpieza y transformación estructural de la serie de tiempo.
    Consume la configuración centralizada para aplicar reglas de negocio.
    """
    def __init__(self, config_path="config.yaml"):
        # Cargar configuración desde YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.params = self.config['preprocessing']
        self.raw_path = self.config['paths']['raw_data']
        self.output_dir = self.config['paths']['cleansed_data']
        self.report_dir = self.config['paths']['reports']
        
        # Estado del reporte de auditoría
        self.report = {
            "phase": "02_Preprocessing",
            "timestamp": datetime.now().isoformat(),
            "input_metrics": {},
            "output_metrics": {},
            "changes": {
                "renamed_columns": {},
                "sentinels_removed": 0,
                "current_month_excluded": False,
                "gaps_filled": 0,
                "values_imputed": 0,
                "columns_added": []
            },
            "data_quality_metrics": {}
        }

    def load_data(self):
        """Carga los datos crudos y estandariza nombres de columnas."""
        print(f"📥 Cargando datos crudos desde: {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        
        # Guardar métricas iniciales
        self.report['input_metrics']['rows'] = len(df)
        self.report['input_metrics']['columns'] = len(df.columns)
        self.report['input_metrics']['column_names'] = df.columns.tolist()
        
        # Estandarización de fecha
        date_col = self.params['time_series']['date_column']
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Renombramiento configurado
        rename_map = self.params.get('rename_columns', {})
        if rename_map:
            df = df.rename(columns=rename_map)
            print(f"🔄 Columnas renombradas: {rename_map}")
            self.report['changes']['renamed_columns'] = rename_map
            
        return df

    def clean_sentinels(self, df):
        """Convierte valores basura (centinelas) a NaN para imputación posterior."""
        target_col = 'unidades'  # Asumimos que ya fue renombrado
        sentinels = self.params['sentinels']['values']
        
        mask = df[target_col].isin(sentinels)
        n_sentinels = mask.sum()
        
        if n_sentinels > 0:
            df.loc[mask, target_col] = np.nan
            print(f"⚠️ Se eliminaron {n_sentinels} valores centinela {sentinels}.")
            
        self.report['changes']['sentinels_removed'] = int(n_sentinels)
        return df

    def flag_outliers(self, df):
        """Identifica outliers usando IQR y crea una bandera binaria (es_atipico)."""
        target_col = 'unidades'
        flag_col = self.params['outliers']['flag_column']
        iqr_mult = self.params['outliers']['iqr_multiplier']
        
        # Verificar si la columna ya existe antes de crearla para el reporte
        cols_before = set(df.columns)
        
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - iqr_mult * IQR
        upper = Q3 + iqr_mult * IQR
        
        # Crear máscara (1 = outlier, 0 = normal)
        # Importante: No borramos, solo marcamos.
        outliers_mask = (df[target_col] < lower) | (df[target_col] > upper)
        df[flag_col] = outliers_mask.astype(int)
        
        # Registrar columna nueva
        cols_after = set(df.columns)
        new_cols = list(cols_after - cols_before)
        if new_cols:
            self.report['changes']['columns_added'].extend(new_cols)
        
        n_outliers = outliers_mask.sum()
        print(f"🚩 Se marcaron {n_outliers} outliers en la columna '{flag_col}'.")
        
        self.report['data_quality_metrics']['outliers_detected'] = int(n_outliers)
        self.report['data_quality_metrics']['iqr_bounds'] = {"lower": round(lower, 2), "upper": round(upper, 2)}
        return df

    def enforce_continuity(self, df):
        """
        Asegura frecuencia mensual (MS), rellena huecos y aplica imputación.
        También filtra el mes actual (Regla del Mes X).
        """
        date_col = self.params['time_series']['date_column']
        target_col = 'unidades'
        flag_col = self.params['outliers']['flag_column']
        
        # 1. Regla del Mes X: Filtrar mes actual si está incompleto
        if self.params['time_series']['exclude_current_month']:
            today = datetime.now()
            current_month_start = datetime(today.year, today.month, 1)
            mask_future = df[date_col] >= current_month_start
            
            n_removed = mask_future.sum()
            if n_removed > 0:
                print(f"✂️ Se truncaron {n_removed} registros del mes actual o futuro.")
                df = df[~mask_future]
            self.report['changes']['current_month_excluded'] = bool(n_removed > 0)

        # 2. Garantizar Frecuencia MS (Month Start)
        df = df.set_index(date_col).sort_index()
        original_len = len(df)
        
        # Re-indexar para cubrir todo el rango sin huecos
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        df = df.reindex(full_idx)
        
        gaps = len(df) - original_len
        if gaps > 0:
            print(f"🧩 Se rellenaron {gaps} huecos temporales en la secuencia.")
            
        # 3. Imputación (Fill Forward / Backfill)
        n_nans_before = df[target_col].isna().sum()
        
        if self.params['imputation']['method'] == 'ffill':
            df[target_col] = df[target_col].ffill()
            
        if self.params['imputation']['fallback_method'] == 'bfill':
            df[target_col] = df[target_col].bfill()
            
        n_nans_after = df[target_col].isna().sum()
        imputed_count = n_nans_before - n_nans_after
        
        print(f"🩹 Se imputaron {imputed_count} valores faltantes (Continuidad/Centinelas).")
        
        # La bandera de outliers en huecos debe ser 0 (no es atípico, es estimado)
        df[flag_col] = df[flag_col].fillna(0).astype(int)
        
        self.report['changes']['gaps_filled'] = int(gaps)
        self.report['changes']['values_imputed'] = int(imputed_count)
        
        return df.reset_index().rename(columns={'index': date_col})

    def save_results(self, df):
        """Guarda el CSV limpio y el reporte JSON."""
        # Métricas finales
        self.report['output_metrics']['rows'] = len(df)
        self.report['output_metrics']['columns'] = len(df.columns)
        self.report['output_metrics']['column_names'] = df.columns.tolist()
        
        # Capturar tipos de datos (convertir a string para JSON)
        self.report['output_metrics']['column_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Clase para codificar tipos no serializables (como int64 de numpy)
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        # CSV
        file_name = self.params['output'].get('filename', 'ventas_preprocesadas.csv') # Fallback por seguridad
        output_path = os.path.join(self.output_dir, file_name)
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"💾 Dataset preprocesado guardado en: {output_path}")
        
        # JSON Report
        report_name = self.params['output'].get('report_name', 'preprocessing_report.json')
        report_path = os.path.join(self.report_dir, report_name)
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        
        print(f"📄 Reporte de auditoría de limpieza guardado en: {report_path}")

    def run_pipeline(self):
        """Ejecuta toda la secuencia de limpieza."""
        print("\n🌊 Iniciando Pipeline de Preprocesamiento (Fase 2)...")
        
        df = self.load_data()
        df = self.clean_sentinels(df)
        df = self.flag_outliers(df)     # Nota: Marcamos outliers ANTES de imputar huecos
        df = self.enforce_continuity(df)
        self.save_results(df)
        
        print("✅ Pipeline de Preprocesamiento finalizado con éxito.")
        return df

if __name__ == "__main__":
    # Prueba unitaria rápida
    processor = DataPreprocessor()
    df_clean = processor.run_pipeline()
    print(df_clean.head())
