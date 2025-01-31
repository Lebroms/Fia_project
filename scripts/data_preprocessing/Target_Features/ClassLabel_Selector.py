import pandas as pd

def classlabel_selector(df):
    """
    Questa funzione divide un dataframe in colonne features e target.
    Il dataframe in ingresso viene sovrascritto dal nuovo dataframe features, mentre
    viene creato un nuovo dataframe target.

    args: DataFrame da dividere.
          colonna_target: lista delle colonne target nel Dataframe.

    return: due dataframe: features, target
    """

    colonne_target = input("\n Quali colonne sono il target (separate da uno spazio): ").split()

    if not colonne_target:
        colonne_target=["classtype_v1"] #colonne target di default del version_1.csv

    for i in colonne_target:
        if i not in df.columns:
            print(f"Errore: la colonna target '{i}' non esiste nel file.")
            return

    df.dropna(subset=colonne_target, inplace=True)
    """
    Questo comando viene aggiunto per agevolare l'utilizzo di funzioni che vengono implementate 
    successivamente. Vengono rimosse tutte le righe contenenti valori NaN all'interno delle
    colonne_target.
    """


    target = df[colonne_target]
    df = df.drop(columns=colonne_target)

    return df, target

