import pandas as pd
from loader.classe_loader import DataLoader


class JsonLoader(DataLoader):
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file Json e lo converte in un DataFrame.

        Args:
            file_path (str): Il percorso del file Json.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """


        print(f"Caricamento del file JSON: {file_path}")
        return pd.read_json(file_path)