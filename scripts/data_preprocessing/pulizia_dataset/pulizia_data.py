# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 17:02:29 2025

@author: emagi
"""
import pandas as pd

"""Si è preferito creare dei metodi statici invece che di istanza per evitare di creare 
    una copia del dataset """

class Df_Processor:
    @staticmethod
    def elimina_colonne(df):
        """
        Rimuove dal DataFrame le colonne specificate nella lista columns_to_drop.
        
        Args:
            df (pd.DataFrame): Il DataFrame di input.
            columns_to_drop (list): Lista con i nomi delle colonne da eliminare. Il default sono
                                   i nomi delle colonne da eliminare del file version_1.csv
        
        Returns:
            pd.DataFrame: Lo stesso DataFrame senza le colonne specificate.
        
        """
        print("\nLe colonne del dataframe caricato sono:\n")
        for col in df.columns:
            print(col)

        columns_to_drop = input("\n Quali vuoi eliminare dall'elenco (separate da uno spazio): ").split()

        if not columns_to_drop:
            columns_to_drop=["Blood Pressure","Sample code number","Heart Rate"] #colonne da eliminare di default del version_1.csv
            
        # Trova le colonne che non esistono nel DataFrame (Differenza tra insiemi)
        missing_columns = list(set(columns_to_drop) - set(df.columns))

        # Se ci sono colonne non presenti genera un errore (probablimente ti sei sbagliato a digitare)
        if missing_columns:
            raise ValueError(f"Le seguenti colonne non esistono nel DataFrame: {missing_columns}")

        # Elimina le colonne 
        return df.drop(columns=columns_to_drop, axis=1)
    
    @staticmethod
    def crea_dummy_variables(df):
        """
        Converte tutte le colonne con valori stringa in variabili dummy con valori 0 o 1.
    
        Args:
            df (pd.DataFrame): Il DataFrame da processare.
    
        Returns:
            pd.DataFrame: Lo stesso DataFrame con variabili dummy sostitutive per le colonne di
            tipo stringa.
        """
        # Identifica le colonne con valori stringa. Pandas rinconosce di default le colonne 
        # contenti stringhe con il tipo object o category. Solo nelle nuove versioni di pandas
        # è stato introdotto il tipo stringa ma è meno usato.
        string_columns = df.select_dtypes(include=['object', 'category']).columns
    
        #  drop_first elimina la prima dummy in ordine alfabetico (del valore). dtype=int mette i valori delle dummies con 0 e 1
        for col in string_columns:
            dummies = pd.get_dummies(df[col], drop_first=True, dtype=int)
            
            if len(dummies.columns) == 1:
                dummies.columns = [f"{col}"]  # Se si ha una sola dummy variable, lascia il nome originale
            
            # Rimuove la colonna originale
            df = df.drop(columns=[col])
            
            # Aggiunge le variabili dummy al DataFrame
            df = pd.concat([df, dummies], axis=1)
    
        return df
    
    @staticmethod
    def gestisci_valori_mancanti(df):
        """
        Gestisce i valori mancanti nel DataSet.

        Parametri:
            df (pd.DataFrame): DataFrame da processare.
            strategy (str): Strategia di gestione ('media', 'mediana', 'moda', 'elimina').

        Uscita:
            pd.DataFrame: stesso DataFrame processato.
        """
        
        strategy=input("\n Scegliere una strategia di sostituzione dei valori mancanti tra le seguenti: \n media \u25CF mediana \u25CF moda \u279C ")
        
        if not strategy:
            strategy="media" #strategia di default
        
        if strategy == "media":
            df.fillna(df.mean(),inplace=True)
        
        elif strategy == "mediana":
            df.fillna(df.median(),inplace=True)

        elif strategy == "moda":
            df.fillna(df.mode().iloc[0],inplace=True)

        elif strategy == "elimina":
            df = df.dropna()
        
        else:
            raise ValueError(f"Strategia '{strategy}' non riconosciuta.")
        
        return df
            
        """
        df.mean restituisce una serie che ha come indici i nomi delle colonne e come
        valori le medie delle rispettive colonne e df.fillna capisce di dover assegnare ad
        ogni valore Nan che incontra la media della sua colonna. Nel caso la colonna contenesse
        valori non numerici sia df.mean che df.fillna lascia i valori Nan inalterati
        """
    
    @staticmethod
    def scala_features(df):
        """
        Esegue il feature scaling su tutte le colonne numeriche di un DataFrame utilizzando normalizzazione o standardizzazione.
        Il datframe deve essere già completamente numerico e non deve contenere Nan

        Args:
            df (pd.DataFrame): Il DataFrame da processare.
            metodo (str): Metodo di scaling, "normalization" o "standardization".

        Returns:
            None: (come sottoprogramma) Viene modificato il DataFrame originale per risparmiare memoria in caso di
            df molto grandi
        """
        
        metodo=input("\n Scegliere un metodo per lo scaling delle feature: \n normalization \u25CF standardization \u279C")
        
        if not metodo:
            metodo="normalization"
        
        for col in df.columns:
            if str(df[col].dtypes) in ['int64', 'float64']: #Controlla il tipo complessivo della colonna (se la colonna
                if metodo == "normalization":         # è mista è di tipo object)
                    # Min-Max Scaling
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                elif metodo == "standardization":
                    # Z-Score Scaling
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df[col] = (df[col] - mean_val) / std_val
                else:
                    raise ValueError(f"Metodo '{metodo}' non riconosciuto. Usa 'normalization' o 'standardization'.")
                    
            else:
                raise ValueError(f"La colonna '{col}' non è numerica e non può essere scalata.")
