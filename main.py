from src.data_loader import DataLoader

def main():
    print("🚀 Iniciando Orquestador de Pronóstico - Buñuelos La Floresta")
    print("📌 Fase Actual: 01_Data_Discovery")
    
    # 1. Instanciar DataLoader y cargar datos con contrato
    try:
        loader = DataLoader()
        df = loader.load_raw_data()
        
        # 2. Ejecutar auditoría completa (incluye outliers)
        print("🔍 Ejecutando auditoría de salud y detección de outliers...")
        health_report = loader.audit_data(df)
        
        # 3. Guardar reporte JSON
        loader.save_report(health_report)
        
        # 4. Generar y guardar figuras
        print("📊 Generando visualizaciones diagnósticas...")
        loader.generate_outlier_plot(df)
        
        print("✅ Fase 1 completada exitosamente.")
        print("🏠 Orquestación finalizada para la Fase 1.")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en la Fase 1: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
