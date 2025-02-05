import pandas as pd
from scripts.interfaccia_utente import interfaccia_utente


# Disabilita il warning SettingWithCopyWarning che veniva generato dalla riga 44 (.replace) perchè non 
# facciamo una copia del df target
pd.set_option('mode.chained_assignment', None)

class Class_label_selector:

    @staticmethod
    def select_label(df, colonne_target):
        """
        Suddivide un DataFrame in due parti: feature set (X) e target (y).
    
        La funzione:
        1. Chiede all'utente di selezionare una o più colonne target.
        2. Rimuove eventuali righe con valori NaN nelle colonne target.
        3. Estrae le colonne target in un nuovo DataFrame.
        4. Se il target contiene solo i valori {2,4}, li converte rispettivamente in {0,1} per una classificazione binaria.
        5. Rimuove le colonne target dal DataFrame originale, restituendo solo le feature.
    
        Args:
            df (pd.DataFrame): Il DataFrame da suddividere.
    
        Returns:
             Due DataFrame:
                - **features (pd.DataFrame)**: Il DataFrame con le colonne delle feature.
                - **target (pd.DataFrame)**: Il DataFrame contenente le colonne target.
    
        """
    
        for i in colonne_target:
            if i not in df.columns:
                print(f"Errore: la colonna target '{i}' non esiste nel file.")
                return
    
        df.dropna(subset=colonne_target, inplace=True)
            
        target = df[colonne_target]
        
        # Convertiamo i 2 in 0 e i 4 in 1 
        if len(colonne_target) == 1:  # Se c'è solo una colonna target
            # Verifica se la colonna contiene solo i valori 2 e 4
            if target[colonne_target[0]].nunique() == 2 and set(target[colonne_target[0]].unique()) == {2, 4}:
                target[colonne_target[0]] = target[colonne_target[0]].replace({2: 0, 4: 1})
        
        # .nunique verifica che la colonna contiene solo due valori
        # .unique verifica che questi due valori sono 2 e 4 (meglio traformarlo in set)
        # .replace sostituisce i 2 e 4 con 0 e 1     
        
        df = df.drop(columns=colonne_target)

        return df, target

