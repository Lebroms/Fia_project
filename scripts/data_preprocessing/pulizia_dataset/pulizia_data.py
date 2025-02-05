# -*- coding: utf-8 -*-
"""
Modulo per la pre-elaborazione dei dati in un DataFrame.

Contiene funzioni statiche per la gestione delle colonne, la codifica di variabili categoriche,
la gestione dei valori mancanti e il feature scaling.

"""
import pandas as pd
from scripts.interfaccia_utente import interfaccia_utente



class Df_Processor:
    """
    Classe per la pre-elaborazione di un DataFrame.

    Tutti i metodi sono statici per evitare la creazione di copie inutili del dataset,
    ottimizzando l'uso della memoria.
    """
    
    @staticmethod
    def elimina_colonne(df,columns_to_drop):
        """
        Rimuove le colonne specificate dall'utente dal DataFrame.

        Args:
            df (pd.DataFrame): Il DataFrame da cui rimuovere le colonne.

        Returns:
            pd.DataFrame: Il DataFrame senza le colonne specificate.

        """
        
            
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
        Converte le colonne categoriche in variabili dummy (one-hot encoding).

        Per ogni colonna categorica, crea nuove colonne binarie rappresentanti le categorie,
        eliminando la colonna originale.

        Args:
            df (pd.DataFrame): Il DataFrame da processare.

        Returns:
            pd.DataFrame: Il DataFrame con variabili dummy al posto delle colonne categoriche.
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
    def gestisci_valori_mancanti(df,strategy):
        """
        Gestisce i valori mancanti nel DataFrame in base alla strategia scelta dall'utente.

        Strategie disponibili:
        - "media": Sostituisce i valori mancanti con la media della colonna.
        - "mediana": Sostituisce i valori mancanti con la mediana della colonna.
        - "moda": Sostituisce i valori mancanti con il valore più frequente della colonna.
        - "elimina": Rimuove tutte le righe contenenti valori mancanti.

        Args:
            df (pd.DataFrame): Il DataFrame da processare.

        Returns:
            pd.DataFrame: Il DataFrame con i valori mancanti gestiti.

        """
        
        
        
        if strategy == "media":
            df.fillna(df.mean(numeric_only=True),inplace=True)
        
        elif strategy == "mediana":
            df.fillna(df.median(numeric_only=True),inplace=True)

        elif strategy == "moda":
            df.fillna(df.mode().iloc[0],inplace=True)
        
        else:
            raise ValueError(f"Strategia '{strategy}' non riconosciuta.")
        
        return df
            
        
    
    @staticmethod
    def scala_features(df,metodo):
        """
        Normalizza o standardizza le colonne numeriche di un DataFrame.

        - "normalization" (Min-Max Scaling): Scala i valori tra 0 e 1.
        - "standardization" (Z-Score Scaling): Centra i valori attorno alla media e li scala in base alla deviazione standard.

        Il DataFrame deve essere completamente numerico e non deve contenere valori NaN.

        Args:
            df (pd.DataFrame): Il DataFrame da processare.

        Returns:
            None: Il DataFrame viene modificato in-place per risparmiare memoria.

        """
        
        
        
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
