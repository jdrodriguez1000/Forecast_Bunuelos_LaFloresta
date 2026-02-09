from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor

def main():
    print("🚀 Iniciando Orquestador de Pronóstico - Buñuelos La Floresta")
    print("📌 Ejecución: Pipeline Fase 1 y Fase 2")
    
    # ---------------------------------------------------------
    # FASE 1: Data Discovery & Auditoría
    # ---------------------------------------------------------
    try:
        print("\n🔎 [FASE 1] Data Discovery & Auditoría Inicial")
        loader = DataLoader() # Lee raw data desde config
        df_raw = loader.load_raw_data()
        
        # Auditoría de salud y outliers (solo observación)
        print("   -> 🏥 Ejecutando chequeo de salud y detección de outliers en crudo...")
        health_report = loader.audit_data(df_raw)
        
        # Guardar artefactos de Fase 1
        loader.save_report(health_report)
        loader.generate_outlier_plot(df_raw)
        print("   ✅ Fase 1 completada. Reportes y gráficos generados.")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en la Fase 1: {e}")
        import traceback
        traceback.print_exc()
        return # Detener si falla la carga inicial

    # ---------------------------------------------------------
    # FASE 2: Preprocessing & Limpieza Estructural
    # ---------------------------------------------------------
    try:
        print("\n🧹 [FASE 2] Preprocessing & Limpieza Estructural")
        
        # Instanciar el procesador que consume config.yaml
        processor = DataPreprocessor()
        
        # Ejecutar el pipeline de limpieza
        # Esto carga de nuevo el raw, aplica reglas y guarda el cleansed
        df_clean = processor.run_pipeline()
        
        print(f"   📊 Resultado Limpieza: {len(df_clean)} registros procesados.")
        print(f"   💾 Archivo limpio guardado en: {processor.output_dir}")
        print("   ✅ Fase 2 completada exitosamente.")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en la Fase 2: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------
    # FASE 3: Análisis Exploratorio & Reglas de Negocio (EDA)
    # ---------------------------------------------------------
    from src.eda_pipeline import run_eda_analysis

    try:
        print("\n📈 [FASE 3] EDA & Validación de Hipótesis")
        print("   -> 🧠 Ejecutando pipeline de análisis exploratorio...")
        print("   -> 📅 Validando hitos (COVID, Retail) y reglas de calendario...")
        
        # Ejecutar el pipeline completo de EDA
        run_eda_analysis()
        
        print("   ✅ Fase 3 completada. Reportes JSON y figuras generados en outputs/.")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en la Fase 3: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------
    # FASE 4: Feature Engineering
    # ---------------------------------------------------------
    from src.features import run_feature_engineering_pipeline

    try:
        print("\n🛠️ [FASE 4] Feature Engineering & Enriquecimiento")
        print("   -> 🗺️ Creando variables de calendario e hitos estructurales...")
        print("   -> 🌍 Integrando indicadores macroeconómicos seleccionados...")
        
        # Ejecutar el pipeline completo de Feature Engineering
        run_feature_engineering_pipeline()
        
        print("   ✅ Fase 4 completada. Dataset .parquet y reporte generados.")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO en la Fase 4: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 Orquestación total finalizada exitosamente.")

if __name__ == "__main__":
    main()

