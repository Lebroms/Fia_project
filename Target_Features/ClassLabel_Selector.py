import pandas as pd

def classlabel_selector(df, colonna_target=["classt"]):
    """
    Questa funzione divide un dataframe in colonne features e target.
    Il dataframe in ingresso viene sovrascritto dal nuovo dataframe features, mentre
    viene creato un nuovo dataframe target.

    args: DataFrame da dividere.
          colonna_target: lista delle colonne target nel Dataframe.

    return: due dataframe: features, target
    """

    for i in colonna_target:
        if i not in df.columns:
            print(f"Errore: la colonna target '{i}' non esiste nel file.")
            return

    # Errore da segnalare in caso non esista un elemento di "colonna target".

    df = df.drop(columns=[colonna_target])
    target = df[[colonna_target]]
    return df, target

