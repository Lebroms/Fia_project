import pandas as pd
from .classe_loader import DataLoader


class XmlLoader(DataLoader):
    """
    Sottoclasse di DataLoader per il caricamento di file xml.

    Questa classe implementa il metodo `load` per leggere file xml in un DataFrame.
    
    """
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file xml in un DataFrame.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: Il contenuto del file caricato in un DataFrame.
        """
        print(f"Caricamento del file XML: {file_path}")
        return pd.read_xml(file_path)
        