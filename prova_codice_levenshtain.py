import pandas as pd
from rapidfuzz.fuzz import ratio
import re

def elimina_colonne_levenshtein(df, columns_to_keep=["clump thickness", 
                                                     "uniformity of cell size",
                                                     "uniformity of cell shape", 
                                                     "marginal adhesion",
                                                     "single epithelial cell size", 
                                                     "bare nuclei",
                                                     "bland chromatin",
                                                     "normal nucleoli",
                                                     "mitoses",
                                                     "class"], threshold=68):
    
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

    # Normalizza le intestazioni del dataframe eliminando gli spazi vuoti e caratteri speciali
    for el in df.columns:
        intestazioni_normalizzate.append(re.sub(r'[^a-zA-Z0-9]', '', el).lower())

    df.columns = intestazioni_normalizzate

    matched_columns = {}

    # Trova i migliori match per le colonne che vogliamo mantenere
    for col in columns_to_keep:
        best_match = max(intestazioni_normalizzate, key=lambda x: ratio(col, x))
        similarity = ratio(col, best_match)  # Ricalcola il grado di similarità
        if similarity >= threshold:
            matched_columns[best_match] = col  # Mappa il nome esistente a quello desiderato

    # Blocca le intestazioni che non sono state abbinate in matched_columns e che iniziano con "col"
    unmatched_columns_to_match = {}

    for el in intestazioni_normalizzate:
        if el.startswith("col") and el not in matched_columns:
            num_str = ''.join(c for c in el if c.isdigit())  # Estrai i numeri

        if num_str and 1 <= int(num_str) <= len(columns_to_keep):  # Controlla validità
            unmatched_columns_to_match[el] = columns_to_keep[int(num_str) - 1]  #assegna il nome corretto
    
    # Unisci matched_columns e unmatched_columns_to_match per rinominare il df
    all_columns_to_rename = {**matched_columns, **unmatched_columns_to_match}

    if not all_columns_to_rename:
        raise ValueError("Nessuna colonna corrispondente trovata con la soglia specificata.")

    # Seleziona solo le colonne presenti nel DataFrame
    valid_columns = list(set(all_columns_to_rename.keys()) & set(df.columns))

    # Filtra e rinomina il DataFrame
    df = df[valid_columns].rename(columns={k: all_columns_to_rename[k] for k in valid_columns})

    return df, matched_columns, unmatched_columns_to_match
