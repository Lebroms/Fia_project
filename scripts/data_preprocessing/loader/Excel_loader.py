import pandas as pd
from .classe_loader import DataLoader

class ExcelLoader(DataLoader):
    """
    Carica un file Excel (.xls o .xlsx) e lo converte in un DataFrame.

    Args:
        file_path (str): Il percorso del file Excel.

    Returns:
        pd.DataFrame: Il dataset caricato come DataFrame.
    """
    
    def load(self, file_path: str) -> pd.DataFrame:
        print(f"Caricamento del file Excel: {file_path}")
        return pd.read_excel(file_path)  # Legge il primo foglio per default
