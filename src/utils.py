import json
import os
import pandas as pd
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(DateTimeEncoder, self).default(obj)

def save_json_report(report_data, file_path):
    """
    Saves a report dictionary to a JSON file.
    """
    import pandas as pd
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False, cls=DateTimeEncoder)
    print(f"Report saved to {file_path}")

def generate_discovery_manifest(df, phase_name="01_data_discovery"):
    """
    Generates a standardized manifest for the discovery phase.
    """
    manifest = {
        "metadata": {
            "project": "Buñuelos La Floresta",
            "phase": phase_name,
            "execution_date": datetime.now().isoformat(),
            "status": "COMPLETED"
        },
        "data_summary": {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df['fecha'].min()),
                "end": str(df['fecha'].max())
            }
        },
        "audit_results": {
            "nulls": int(df.isnull().sum().sum()),
            "duplicates_rows": int(df.duplicated().sum()),
            "duplicates_dates": int(df['fecha'].duplicated().sum())
        },
        "statistics": df.describe().to_dict()
    }
    return manifest
