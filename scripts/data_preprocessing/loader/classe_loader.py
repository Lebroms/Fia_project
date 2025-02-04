from abc import ABC, abstractmethod
import pandas as pd

# Interfaccia comune per tutti i loader
class DataLoader(ABC):
    """
    Classe astratta che definisce un'interfaccia per il caricamento di dati da file.

    Le classi concrete che ereditano da DataLoader devono implementare il metodo `load`, 
    che carica i dati da un file specificato e restituisce un DataFrame.

    Metodi:
        load(file_path: str) -> pd.DataFrame:
            Metodo astratto che deve essere implementato nelle sottoclassi per leggere 
            e restituire un DataFrame a partire dal file specificato.
    """
    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica i dati da un file e li restituisce come un DataFrame.

        Questo metodo deve essere implementato dalle classi figlie.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
        
        pass