# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:16:48 2025

@author: emagi
"""

import pandas as pd

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