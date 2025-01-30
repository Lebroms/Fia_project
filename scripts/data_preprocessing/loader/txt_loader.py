import pandas as pd
from .classe_loader import DataLoader

class TxtLoader(DataLoader):
    """
    Sottoclasse di DataLoader specializzata nel caricamento di file TXT.
    Il metodo load verifica il contenuto del file per scegliere il separatore più adatto.

    Args:
        file_path (str): Il percorso del file da caricare.

    Returns:
        pd.DataFrame: I dati caricati come DataFrame.
    """

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file di testo e lo converte in un DataFrame in base al separatore rilevato.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: I dati caricati come DataFrame.
        """
        # Imposta un separatore di default
        separator = None

        # Legge la prima riga per capire il formato
        with open(file_path, "r", encoding="utf-8") as file:
            first_line = file.readline()

            # Controlla i separatori più comuni
            if "\t" in first_line:
                separator = "\t"  # File TSV
            elif ";" in first_line:
                separator = ";"
            elif "," in first_line:
                separator = ","
            else:
                separator = None  # Nessun separatore rilevato, il file verrà letto come testo singolo

        print(f"Caricamento del file TXT: {file_path} con separatore '{separator}'")

        # Se non viene rilevato un separatore, il file viene letto come testo grezzo in una sola colonna
        if separator:
            return pd.read_csv(file_path, sep=separator)
        else:
            return pd.DataFrame({"text": open(file_path, "r", encoding="utf-8").readlines()})
