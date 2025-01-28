# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:13:44 2025

@author: emagi
"""


def gestisci_valori_mancanti(df, strategy="media"):
    """
    Gestisce i valori mancanti nel DataSet.

    Parametri:
        df (pd.DataFrame): DataFrame da processare.
        strategy (str): Strategia di gestione ('media', 'mediana', 'moda', 'elimina').

    Uscita:
        pd.DataFrame: DataFrame processato.
    """
    if strategy == "media":
        return df.fillna(df.mean())  
    elif strategy == "mediana":
        return df.fillna(df.median())
    elif strategy == "moda":
        return df.fillna(df.mode().iloc[0])
    elif strategy == "elimina":
        return df.dropna()
    else:
        raise ValueError(f"Strategia '{strategy}' non riconosciuta.")
        
        """df.mean restituisce una serie che ha come indici i nomi delle colonne e come
        valori le medie delle rispettive colonne e df.fillna capisce di dover assegnare ad
        ogni valore Nan che incontra la media della sua colonna. Nel caso la colonna contenesse
        valori non numerici si df.mean che df.fillna lascia i valori Nan inalterati"""
        
        
