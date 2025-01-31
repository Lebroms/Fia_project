# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:13:44 2025

@author: emagi
"""
import pandas as pd

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
    

    
    else:
        raise ValueError(f"Strategia '{strategy}' non riconosciuta.")
    
    return df
        
    """
    df.mean restituisce una serie che ha come indici i nomi delle colonne e come
    valori le medie delle rispettive colonne e df.fillna capisce di dover assegnare ad
    ogni valore Nan che incontra la media della sua colonna. Nel caso la colonna contenesse
    valori non numerici sia df.mean che df.fillna lascia i valori Nan inalterati
    """
        
        

