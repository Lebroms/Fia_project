import pandas as pd

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
    nome_file = input("Inserire il nome del file in cui salvare le performance (Invio per usare 'performance_model.xlsx'): ").strip()
    
    # Se l'utente preme solo Invio, usa il nome di default
    if nome_file == "":
        nome_file = "performance_model.xlsx"
        print(f"Nessun file inserito. Carico i risultati nel file {nome_file}")

    # Assicuriamoci che il nome del file abbia l'estensione corretta
    if not nome_file.endswith(".xlsx"):
        nome_file += ".xlsx"

    # **Percorso fisso della cartella 'results' dentro 'Fia_project'**
    percorso_completo = f"results/{nome_file}"  # Salva direttamente in 'results/'

    try:
        # Controlliamo se il file esiste
        pd.read_excel(percorso_completo)
        file_esiste = True
    except FileNotFoundError:
        file_esiste = False

    # Se il file esiste, chiediamo all'utente cosa fare
    if file_esiste:
        scelta = input(f"Il file '{percorso_completo}' esiste già. Vuoi sovrascriverlo? (s/n): ").strip().lower()
        
        if scelta == "n":
            nuovo_nome = input("Inserisci il nuovo nome del file (es. nuovo_file.xlsx): ").strip()
            if not nuovo_nome.endswith(".xlsx"):
                nuovo_nome += ".xlsx"  # Assicuriamoci che il file abbia l'estensione giusta
            percorso_completo = f"results/{nuovo_nome}"  # Usa il nuovo nome scelto
    

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
    print(f"Il file si trova nella cartella 'results' dentro 'Fia_project'.")

