# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:39:40 2025

@author: emagi
"""

import pandas as pd

def crea_dummy_variables(df):
    """
    Converte tutte le colonne con valori stringa in variabili dummy con valori 0 o 1.

    Args:
        df (pd.DataFrame): Il DataFrame da processare.

    Returns:
        pd.DataFrame: Lo stesso DataFrame con variabili dummy sostitutive per le colonne di
        tipo stringa.
    """
    # Identifica le colonne con valori stringa. Pandas rinconosce di default le colonne 
    # contenti stringhe con il tipo object o category. Solo nelle nuove versioni di pandas
    # è stato introdotto il tipo stringa ma è meno usato.
    
    string_columns = df.select_dtypes(include=['object', 'category']).columns

    # Crea le dummy variables. Il parametro columns accetta qualsiasi iterabile quindi anche
    # un index proveniente da df.columns. drop_first elimina la prima dummy in ordine 
    # alfabetico (del valore). dtype=int metti i valori delle dummies con 0 e 1

    for col in string_columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        dummies.columns = [f"{col} {i}" for i in range(len(dummies.columns))]

        if len(dummies.columns) == 1:
            dummies.columns=[f"{col}"] #se si ha una sola dummy variable lasciale il suo nome originale
                                       # (e non {col} 0)
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)

        """
        Questo ciclo for itera su tutte le colonne che contengono valori stringa e crea
        un nuovo dataframe (dummies) che contiene solo le colonne dummies che  
        vengono nominate come <col>, <col> 1, <col> 2, ecc. 
        """

    return df






