import pandas as pd
import os
from src.utils import generate_discovery_manifest, save_json_report

def main():
    print("🚀 Iniciando Orquestador de Pronóstico - Buñuelos La Floresta")
    print("📌 Fase Actual: 01_Data_Discovery")
    
    # 1. Carga de datos crudos
    raw_path = 'data/01_raw/ventas_mensuales.csv'
    if not os.path.exists(raw_path):
        print(f"❌ Error: No se encuentra el archivo {raw_path}")
        return

    df = pd.read_csv(raw_path)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # 2. Generación de Manifiesto de Descubrimiento
    print("🔍 Ejecutando auditoría de salud de datos...")
    manifest = generate_discovery_manifest(df)
    
    # 3. Guardar reporte informativo
    report_path = 'outputs/reports/discovery_report.json'
    save_json_report(manifest, report_path)
    
    print(f"✅ Reporte de Fase 1 generado exitosamente en: {report_path}")
    print("🏠 Orquestación finalizada para la Fase 1.")

if __name__ == "__main__":
    main()
