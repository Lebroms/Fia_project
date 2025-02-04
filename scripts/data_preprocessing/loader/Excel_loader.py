import pandas as pd
from .classe_loader import DataLoader

class ExcelLoader(DataLoader):
    """
    Sottoclasse di DataLoader per il caricamento di file Excel.

    Questa classe implementa il metodo `load` per leggere file Excel in un DataFrame.
    
    """
    
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file Excel in un DataFrame.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: Il contenuto del file caricato in un DataFrame.
        """
        print(f"Caricamento del file Excel: {file_path}")
        return pd.read_excel(file_path)  # Legge il primo foglio per default
