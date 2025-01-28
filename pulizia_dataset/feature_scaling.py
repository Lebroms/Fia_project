# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:27:07 2025

@author: emagi
"""

import pandas as pd

def scala_features(df, metodo="normalization"):
    """
    Esegue il feature scaling su tutte le colonne numeriche di un DataFrame utilizzando normalizzazione o standardizzazione.

    Args:
        df (pd.DataFrame): Il DataFrame da processare.
        metodo (str): Metodo di scaling, "normalization" o "standardization".

    Returns:
        None: Viene modificato il DataFrame originale per norisparmiare memoria in caso di df molto grandi
    """
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']: #Controlla il tipo complessivo della colonna (se la colonna è mista è di tipo object)
            if metodo == "normalization":
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


