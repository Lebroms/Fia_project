import pandas as pd



class interfaccia_utente():



    @staticmethod    
    def get_columns_to_drop_input(df):
        print("\nLe colonne del dataframe caricato sono:\n")
        for col in df.columns:
            print(col)

        columns_to_drop = input("\n Quali vuoi eliminare dall'elenco (separate da uno spazio): ").split()

        if not columns_to_drop:
            columns_to_drop=["Blood Pressure","Sample code number","Heart Rate"] #colonne da eliminare di default del version_1.csv
        
        return columns_to_drop
    
    @staticmethod
    def get_replacement_stretegy():
        strategy=input("\n Scegliere una strategia di sostituzione dei valori mancanti tra le seguenti: \n media \u25CF mediana \u25CF moda \u279C ")
        
        if not strategy:
            strategy="media" #strategia di default
        return strategy
    
    @staticmethod
    def get_scaling_method():
        method=input("\n Scegliere un metodo per lo scaling delle feature: \n normalization \u25CF standardization \u279C")
        
        if not method:
            method="normalization"
        return method
    
    @staticmethod
    def get_target_columns():
        colonne_target = input("\n Quali colonne sono il target (separate da uno spazio): ").split()

        if not colonne_target:
            colonne_target=["classtype_v1"] #colonne target di default del version_1.csv
        
        return colonne_target
    

    @staticmethod
    def get_validation_method():
        
        print("\n Scegliere quale delle seguenti strategia di validazione usare: \n")
        print("\u25BA Premi \U00000031\U0000FE0F\U000020E3 per Holdout \n")
        print("\u25BA Premi \U00000032\U0000FE0F\U000020E3 per Random Sub Sampling \n")
        print("\u25BA Premi \U00000033\U0000FE0F\U000020E3 per K-fold Cross Validation \n")

        strategy = input("Inserisci il numero della strategia scelta: ").strip()

        return strategy
    
    @staticmethod
    def get_size_of_test():
        while True:  # Ciclo per chiedere il valore di test_size finché non è valido
            test_size = input(
                "Imposta la percentuale di campioni del dataset da assegnare al test set (valore tra 0 e 1): ").strip()

            if not test_size:  # Se l'utente preme Invio senza inserire nulla
                test_size = "0.2"  # Imposta il valore predefinito
                print("Nessun valore inserito. Imposto test_size a 0.2 di default.")

            test_size = test_size.replace(",", ".")  # Sostituisce la virgola con il punto

            try:
                test_size = float(test_size)  # Converte in float
                if 0 < test_size < 1:
                    break  # Esce dal ciclo se il valore è valido
                else:
                    print("Errore: Il valore deve essere compreso tra 0 e 1. Riprova.")
            except ValueError:
                print("Errore: Inserisci un numero valido (es. 0.2 o 0,2). Riprova.")
        return test_size
    

    @staticmethod
    def get_k_neighbours():
        '''
        Funzione che permette all'utente di scegliere il numero di k vicini da utilizzare per il 
        Classificatore. Gestisce vari casi di input stampando messaggi di errore e chiedendo di reinserire 
        k nei casi in cui il valore di k inserito non vada bene.

        Return:
        k= intero che rappresenta il numero di vicini da usare per il Classificatore
        '''


        while True:
            k = input(
                "Inserire il valore dei k vicini da voler usare per costruire il Classificatore KNN (default: 3): ").strip()

            if k == "":  # Se l'utente non inserisce nulla
                k = 3  # Imposta il valore di default
                print("Nessun valore inserito. Impostato k = 3 di default.")
                break

            try:
                k = int(k)
                if k <= 0:
                    print("Errore: k deve essere un intero positivo. Riprova.")

                else:
                    break
            except ValueError:
                print("Errore: Inserisci un numero intero valido. Riprova.")

        print(f"Impostato il numero di vicini k = {k}")

        print("\nCalcolando la predizione...")

        return k
    
    @staticmethod
    def get_num_experiments():
        while True:  # Ciclo per chiedere il valore di num_experiments finché non è valido
            num_experiments = input("Imposta il numero di esperimenti da eseguire (numero intero positivo)")

            if not num_experiments:  # Se l'utente preme Invio senza inserire nulla
                num_experiments = "10"  # Imposta il valore predefinito
                print("Nessun valore inserito. Imposto numero esperimenti a 10 di default.")


            try:
                num_experiments = int(num_experiments)
                if num_experiments > 0:
                    break  # Esce dal ciclo se il valore è valido
                else:
                    print("Errore: Il valore deve essere positivo. Riprova.")
            except ValueError:
                print("Errore: Inserisci un numero valido (es. 10). Riprova.")
        return int(num_experiments)
  
    @staticmethod
    def get_num_folds():
        while True:  # Ciclo per chiedere il valore di num_folds finché non è valido
            n_folds = input(
                "Imposta il numero di fold in cui dividere il dataset (numero intero ≥ 2): ").strip()

            if not n_folds:  # Se l'utente preme Invio senza inserire nulla
                n_folds = "10"  # Imposta il valore predefinito
                print("Nessun valore inserito. Imposto numero fold a 10 di default.")

            try:
                n_folds = int(n_folds)
                if n_folds >= 2:
                    break  # Esce dal ciclo se il valore è valido
                else:
                    print("Errore: Il valore deve essere un numero intero maggiore o uguale a 2. Riprova.")
            except ValueError:
                print("Errore: Inserisci un numero intero valido (es. 10). Riprova.")
        return n_folds
    
    @staticmethod
    def get_metrics_to_calculate():
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

    @staticmethod
    def get_mod_calculation_metrics(num):
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
                    

    @staticmethod
    def get_file():
        nome_file = input("Inserire il nome del file in cui salvare le performance (Invio per usare 'performance_model.xlsx'): ").strip()
    
        # Se l'utente preme solo Invio, usa il nome di default
        if nome_file == "":
            nome_file = "performance_model.xlsx"
            print(f"Nessun file inserito. Carico i risultati nel file {nome_file}")

        # Assicuriamoci che il nome del file abbia l'estensione corretta
        if not nome_file.endswith(".xlsx"):
            nome_file += ".xlsx"

        # **Percorso fisso della cartella 'results' dentro 'Fia_project'**
        percorso_completo = f"risultati/{nome_file}"  # Salva direttamente in 'results/'

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
                percorso_completo = f"risultati/{nuovo_nome}"  # Usa il nuovo nome scelto
        return percorso_completo

