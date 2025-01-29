# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:26:26 2025

@author: emagi
"""

import pandas as pd
from rapidfuzz.fuzz import ratio
import re

def elimina_colonne_levenshtein(df, columns_to_keep=["clump thickness 1", 
                                                     "uniformity of cell size 2",
                                                     "uniformity of cell shape 3", 
                                                     "marginal adhesion 4",
                                                     "single epithelial cell size 5", 
                                                     "bare nuclei 6",
                                                     "bland chromatin 7",
                                                     "normal nucleoli 8",
                                                     "mitoses 9",
                                                     "classt 10"], threshold=68):
    
    """
    Mantiene nel dataset solo le colonne che hanno una corrispondenza approssimativa 
    usando la distanza di Levenshtein.
    
    Parametri:
    df: pandas DataFrame
    columns_to_keep: list[str], elenco delle colonne desiderate
    threshold: int, soglia di similarità (default 80 su 100)
    
    return: lo stesso DataFrame processato
    """

    intestazioni_normalizzate=[]

    #normalizza le intestazioni del dataframe per la funzione successiva 
    #eliminando gli spazi vuoti
    for el in df.columns:
        intestazioni_normalizzate.append(re.sub(r'[^a-zA-Z0-9]', '', el).lower())


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
    df=df[list(matched_columns.keys())].rename(columns=matched_columns)
    
    return df, matched_columns



            
        
            