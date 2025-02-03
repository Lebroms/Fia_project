

def scegli_modalita_calcolo_metriche(num):
    '''
    Funzione che permette di scegliere la modalità di calcolo:fare la media delle singole metriche 
    calcolate per il numero di esperimenti selezionato, oppure restituire le  metriche per i singoli 
    esperimenti 

    Parametri: 
    num= intero che rappresenta il numero di esperimenti 

    Return:
    Modalità= variabile booleana che viene impostata su True se si vuole fare la media 
    delle varie metriche su tutti gli esperimenti 
    '''

    while True:
        print("Scegli la modalità di calcolo delle metriche:")
        print(f"1. Fai la media delle metriche selezionate sui {num} esperimenti")
        print("2. Calcola le metriche selezionate per i singoli esperimenti")
        
        scelta = input().strip()
        
        if scelta == "1":
            modalita = True
            break
        elif scelta == "2":
            modalita = False
            break
        else:
            print("Scelta non valida. Per favore, inserisci 1 o 2.")
    
    return modalita



def scegli_metriche():
        """
        Funzione che chiede all'utente di selezionare le metriche da calcolare e restituisce la lista delle scelte.

        Return:
        metriche_scelte= lista numerica delle metriche selezionate 
        """
        lista_metriche = [
        "Accuracy", "Error Rate", "Sensitivity", "Specificity",
        "Geometric Mean", "Area Under the Curve", "Tutte le metriche"]

        n = len(lista_metriche)
        numeri_validi = {str(i) for i in range(1, n+1)}  # Numeri validi (da "1" a "7")

        while True:
            print("\nScegliere tra le seguenti metriche quelle che si vogliono calcolare:\n")
            for index, el in enumerate(lista_metriche, start=1):
                print(f"\u25BA {index}. Per selezionare {el} premere {index}\n")

            metriche_scelte = input("Inserisci i numeri delle metriche separati da spazio: ").split()

            # Se l'utente non inserisce nulla, seleziona tutte le metriche
            if not metriche_scelte:
                metriche_scelte = ["7"] 

            # Controlla se tutte le metriche scelte sono valide
            if all(el in numeri_validi for el in metriche_scelte):
                return metriche_scelte  # Se sono tutte valide, restituisce la lista
            else:
                print("\nErrore: Alcuni numeri inseriti non sono validi. Riprova.")