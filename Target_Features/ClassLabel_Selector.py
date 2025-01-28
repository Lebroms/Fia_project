import pandas as pd

def classlabel_selector(df, colonna_target):
    """
    Questa funzione divide un file dataset CSV in colonne features e target
    e salva i risultati in file separati. La colonna target contiene i valori
    beningno/maligno (2 o 4), le altre colonne contengono i valori restanti.

    input_file: Path del file CSV in ingresso.
    colonna_target: Nome della colonna target nel dataset.
    """

    if colonna_target not in df.columns:
        print(f"Errore: la colonna target '{colonna_target}' non esiste nel file.")
        return

    # Errore da segnalare in caso non esista la colonna target.

    features = df.drop(columns=[colonna_target])
    target = df[[colonna_target]]
    return features, target

    # Divide il dataset in features e target.


