import pandas as pd
from .classe_loader import DataLoader

class TxtLoader(DataLoader):
    """
    Sottoclasse di DataLoader per il caricamento di file di testo (.txt).

    Questa classe implementa il metodo `load` per leggere file di testo strutturati 
    (con separatori come tabulazione, punto e virgola o virgola) o file di testo non 
    strutturati.

    Il metodo determina automaticamente il separatore in base al contenuto della prima 
    riga del file o, se nessun separatore è presente, lo carica come un'unica colonna `"text"`.
    """

    def load(self, file_path: str) -> pd.DataFrame:
        """
        Carica un file di testo in un DataFrame, determinando automaticamente il formato.

        Il comportamento è il seguente:
        - Se la prima riga contiene un separatore (`\t`, `;`, `,`), il file viene trattato
          come tabellare e letto con `pd.read_csv()` usando il separatore rilevato.
        - Se nessun separatore viene trovato, il file viene caricato come una colonna unica
          denominata `"text"`.

        Args:
            file_path (str): Il percorso del file da caricare.

        Returns:
            pd.DataFrame: Il contenuto del file caricato in un DataFrame.''' 

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
