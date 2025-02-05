import pandas as pd
from scripts.interfaccia_utente import interfaccia_utente


def salva_metriche_su_excel(lista_dizionari,percorso_completo):
    """
    Salva una lista di dizionari di metriche in un file Excel.  

    Args:
        lista_dizionari (list): Lista di dizionari contenenti metriche calcolate.

    Struttura del file Excel salvato:
        - Se è presente un solo dizionario, le metriche sono salvate in due colonne: 
          la prima per il nome della metrica e la seconda per il valore.
        - Se ci sono più dizionari (uno per ogni esperimento), il file avrà una struttura con:
          - Le metriche disposte sulle righe.
          - Ogni esperimento in una colonna separata.

    Returns:
        None: Il file viene salvato e non viene restituito alcun valore.

    Output:
        - Il file Excel viene generato e salvato nel percorso specificato.
        - Un messaggio di conferma viene stampato a schermo con il percorso del file salvato.
    """
    
    

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

