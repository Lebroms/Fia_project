import pandas as pd
from loader.classe_loader import DataLoader

class XmlLoader(DataLoader):
    """
        Carica un file XML e lo converte in un DataFrame.

        Args:
            file_path (str): Il percorso del file XML.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
    def load(self, file_path: str) -> pd.DataFrame:
        print(f"Caricamento del file XML: {file_path}")
        return pd.read_xml(file_path)
        