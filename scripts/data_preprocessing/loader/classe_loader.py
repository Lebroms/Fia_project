from abc import ABC, abstractmethod
import pandas as pd

# Interfaccia comune per tutti i loader
class DataLoader(ABC):
    """
        Metodo astratto per caricare un file.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        
        pass