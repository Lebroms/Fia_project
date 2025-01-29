# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:13:44 2025

@author: emagi
"""
import pandas as pd

def gestisci_valori_mancanti(df, strategy="media"):
    """
    Gestisce i valori mancanti nel DataSet.

    Parametri:
        df (pd.DataFrame): DataFrame da processare.
        strategy (str): Strategia di gestione ('media', 'mediana', 'moda', 'elimina').

    Uscita:
        pd.DataFrame: stesso DataFrame processato.
    """
    if strategy == "media":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    elif strategy == "mediana":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    elif strategy == "moda":
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])

    elif strategy == "elimina":
        df = df.dropna()
    
    else:
        raise ValueError(f"Strategia '{strategy}' non riconosciuta.")
    
    return df
        
    """df.mean restituisce una serie che ha come indici i nomi delle colonne e come
        valori le medie delle rispettive colonne e df.fillna capisce di dover assegnare ad
        ogni valore Nan che incontra la media della sua colonna. Nel caso la colonna contenesse
        valori non numerici sia df.mean che df.fillna lascia i valori Nan inalterati"""
        
        

