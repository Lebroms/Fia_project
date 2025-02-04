import pandas as pd
from scripts.interfaccia_utente import interfaccia_utente


def salva_metriche_su_excel(lista_dizionari):
    """
    Chiede all'utente il nome del file Excel e salva una lista di dizionari di metriche
    nella cartella 'results' dentro 'Fia_project'.
    Se l'utente preme Invio senza inserire un nome, viene usato un nome di default.

    Args:
        lista_dizionari (list): Lista di dizionari contenenti metriche.

    Returns:
        None: Il file Excel viene salvato nella cartella 'results' dentro 'Fia_project'.
    """

    # Chiede il nome del file all'utente
    percorso_completo=interfaccia_utente.get_file()
    

    if len(lista_dizionari) == 1:
        df = pd.DataFrame(lista_dizionari[0].items(), columns=["", "Valore"])
        df = df.set_index("")  # Imposta le metriche come indice
    else:
        # Caso 2: Più dizionari (uno per ogni esperimento)
        df = pd.DataFrame(lista_dizionari).T  # Trasponiamo per avere metriche sulle righe
        df.columns = [f"Exp {i+1}" for i in range(len(lista_dizionari))]  # Rinomina le colonne


    # **Salvataggio effettivo nel percorso corretto**
    df.to_excel(percorso_completo, index=True)

    print(f"Le metriche sono state salvate in '{percorso_completo}'\n")
    print(f"Il file si trova nella cartella 'risultati' dentro 'Fia_project'.")

