import pandas as pd
from .classe_loader import DataLoader

class CsvLoader(DataLoader):
    """
    Sottoclasse di DataLoader per il caricamento di file CSV e TSV.

    Questa classe implementa il metodo `load` per leggere file CSV e TSV in un DataFrame.
    Il metodo determina automaticamente il separatore in base all'estensione del file
    o analizzando la prima riga del contenuto.
    """

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file CSV o TSV in un DataFrame, determinando automaticamente il separatore.

        Il separatore viene scelto come segue:
        - Se il file ha estensione `.tsv`, viene usato il separatore di tabulazione (`\t`).
        - Se il file ha estensione `.csv`, viene analizzata la prima riga:
          - Se contiene `;`, viene usato `;` come separatore.
          - Altrimenti, viene usata la virgola `,` come separatore predefinito.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: Il contenuto del file caricato in un DataFrame.
        """
        # Imposta un separatore di default
        separator = ','

        # Se il file è un TSV, usa il separatore di tabulazione
        if file_path.endswith('.tsv'):
            separator = '\t'
        elif file_path.endswith('.csv'):
            # Controlla se potrebbe essere un CSV con ';' come separatore
            with open(file_path, "r", encoding="utf-8") as file:
                first_line = file.readline()
                if ";" in first_line:
                    separator = ";"

        
        return pd.read_csv(file_path, sep=separator)
