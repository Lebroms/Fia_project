# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 19:54:06 2025

@author: emagi
"""

import pandas as pd

def convert_comma_to_dot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte tutte le colonne di tipo 'object' che contengono numeri con la virgola 
    in numeri con il punto e le trasforma in float, se possibile.
    
    Args:
        df (pd.DataFrame): Il DataFrame da modificare.
    
    Returns:
        pd.DataFrame: Il DataFrame con i valori corretti.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace(',', '.', regex=True)

        # Prova a convertire in float, se tutti i valori possono essere convertiti
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # Se ci sono errori, mantiene il tipo object  

    return df  
