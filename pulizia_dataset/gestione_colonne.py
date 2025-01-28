# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:26:26 2025

@author: emagi
"""

import pandas as pd
from rapidfuzz.fuzz import ratio
import re




def elimina_colonne_levenshtein(df, columns_to_keep, threshold=80):
    """
    Mantiene nel dataset solo le colonne che hanno una corrispondenza approssimativa 
    usando la distanza di Levenshtein.
    
    Parametri:
    df: pandas DataFrame
    columns_to_keep: list[str], elenco delle colonne desiderate
    threshold: int, soglia di similarità (default 80 su 100)
    
    :return: pandas DataFrame
    """

    
    colonne_df = df.columns.tolist()

    intestazioni_normalizzate=[]

    #normalizza le intestazioni del dataframe per la funzione successiva 
    #eliminando gli spazi vuoti
    for el in colonne_df:
        intestazioni_normalizzate.append(re.sub(r'[^a-zA-Z]', '', el).lower())

    





    df.columns = intestazioni_normalizzate

    matched_columns = {}

    for col in columns_to_keep:
        best_match = max(intestazioni_normalizzate, key=lambda x: ratio(col, x))
        similarity = ratio(col, best_match) #ricalcola il grado di similarità tra best match e col per confrontarlo con il threshold impostato
        if similarity >= threshold:
            matched_columns[best_match] = col  # Mappa il nome esistente a quello desiderato
            

    if not matched_columns:
        raise ValueError("Nessuna colonna corrispondente trovata con la soglia specificata.")


    # Rinomina e restituisce il dataset filtrato
    df_filtered = df[list(matched_columns.keys())].rename(columns=matched_columns)
    return df_filtered


            
        
            