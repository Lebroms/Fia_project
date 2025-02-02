import pandas as pd

# Disabilita il warning SettingWithCopyWarning che veniva generato dalla riga 37 perchè non 
# facciamo una copia del df target
pd.set_option('mode.chained_assignment', None)

def classlabel_selector(df, colonne_target=["classtype_v1"]):
    """
    Questa funzione divide un dataframe in colonne features e target.
    Il dataframe in ingresso viene sovrascritto dal nuovo dataframe features, mentre
    viene creato un nuovo dataframe target.

    args: DataFrame da dividere.
          colonna_target: lista delle colonne target nel Dataframe.

    return: due dataframe: features, target
    """

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
    
    # Convertiamo i 2 in 0 e i 4 in 1 
    if len(colonne_target) == 1:  # Se c'è solo una colonna target
        # Verifica se la colonna contiene solo i valori 2 e 4
        if target[colonne_target[0]].nunique() == 2 and set(target[colonne_target[0]].unique()) == {2, 4}:
            target[colonne_target[0]] = target[colonne_target[0]].replace({2: 0, 4: 1})
    
    # .nunique verifica che la colonna contiene solo due valori
    # .unique verifica che questi due valori sono 2 e 4 (meglio traformarlo in set)        
    
    df = df.drop(columns=colonne_target)

    return df, target

