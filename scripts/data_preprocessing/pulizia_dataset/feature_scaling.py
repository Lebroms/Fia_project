
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:27:07 2025

@author: emagi
"""

import pandas as pd

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


