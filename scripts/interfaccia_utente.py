import pandas as pd



class interfaccia_utente():

    """
    Classe che fornisce metodi statici per interagire con l'utente, consentendo di selezionare
    parametri per la pulizia dei dati, la validazione e la valutazione del modello.
    """



    @staticmethod    
    def get_columns_to_drop_input(df):

        """
        Mostra le colonne disponibili nel DataFrame e chiede all'utente quali eliminare.
        Se non viene inserito alcun input, vengono rimosse le colonne di default.

        Args:
            df (pd.DataFrame): Il DataFrame da cui selezionare le colonne da eliminare.

        Returns:
            list: Lista delle colonne da eliminare.
        """
        print("\nLe colonne del dataframe caricato sono:\n")
        for col in df.columns:
            print(col)

        columns_to_drop = input("\n Quali vuoi eliminare dall'elenco (separate da un trattino): ").split("//")

        if columns_to_drop == ['']:
            columns_to_drop=["Blood Pressure","Sample code number","Heart Rate"] #colonne da eliminare di default del version_1.csv
        
        return columns_to_drop
    
    @staticmethod
    def get_replacement_stretegy():
        """
        Chiede all'utente di scegliere una strategia per la sostituzione dei valori mancanti.
        Se non viene inserito alcun input, utilizza la strategia "media" di default.

        Returns:
            str: Strategia scelta ("media", "mediana" o "moda").
        """
        strategy=input("\n Scegliere una strategia di sostituzione dei valori mancanti tra le seguenti: \n media \u25CF mediana \u25CF moda \u279C ")
        
        if not strategy:
            strategy="media" #strategia di default
        return strategy
    
    @staticmethod
    def get_scaling_method():
        """
        Chiede all'utente di scegliere un metodo per lo scaling delle feature.
        Se non viene inserito alcun input, utilizza "normalization" di default.

        Returns:
            str: Metodo di scaling scelto ("normalization" o "standardization").
        """
        method=input("\n Scegliere un metodo per lo scaling delle feature: \n normalization \u25CF standardization \u279C")
        
        if not method:
            method="normalization"
        return method
    
    @staticmethod
    def get_target_columns():
        """
        Chiede all'utente quali colonne sono il target.
        Se non viene inserito alcun input, utilizza "classtype_v1" di default.

        Returns:
            list: Lista delle colonne target selezionate.
        """
        colonne_target = input("\n Quali colonne sono il target (separate da uno spazio): ").split()

        if not colonne_target:
            colonne_target=["classtype_v1"] #colonne target di default del version_1.csv
        
        return colonne_target
    

    @staticmethod
    def get_validation_method():
        """
        Chiede all'utente quale strategia di validazione utilizzare.

        Returns:
            str: Numero della strategia scelta dall'utente.
        """
        
        print("\n Scegliere quale delle seguenti strategia di validazione usare: \n")
        print("\u25BA Premi \U00000031\U0000FE0F\U000020E3  per Holdout \n")
        print("\u25BA Premi \U00000032\U0000FE0F\U000020E3  per Random Sub Sampling \n")
        print("\u25BA Premi \U00000033\U0000FE0F\U000020E3  per K-fold Cross Validation \n")

        strategy = input("Inserisci il numero della strategia scelta: ").strip()

        return strategy
    
    @staticmethod
    def get_size_of_test():
        """
        Chiede all'utente la percentuale di dati da destinare al test set.
        Se non viene inserito alcun valore, imposta di default 0.2.

        Returns:
            float: Percentuale del dataset assegnata al test set.
        """
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
        """
        Chiede all'utente di selezionare il numero di vicini per il classificatore KNN.
        Se non viene inserito alcun valore, imposta di default k=3.

        Returns:
            int: Numero di vicini da usare nel KNN.
        """


        while True:
            k = input(
                "Inserire il valore dei k vicini da voler usare per costruire il Classificatore KNN (default: 10): ").strip()

            if k == "":  # Se l'utente non inserisce nulla
                k = 10  # Imposta il valore di default
                print("Nessun valore inserito. Impostato k = 10 di default.")
                break

            try:
                k = int(k)
                if k <= 0:
                    print("Errore: k deve essere un intero positivo. Riprova.")

                else:
                    break
            except ValueError:
                print("Errore: Inserisci un numero intero valido. Riprova.")

        

        return k
    
    @staticmethod
    def get_num_experiments():
        """
        Chiede all'utente il numero di esperimenti da eseguire per la validazione.
        Se non viene inserito alcun valore, imposta il valore di default a 10.

        Returns:
            int: Numero di esperimenti da eseguire.
        """
        
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
        """
        Chiede all'utente il numero di fold per la validazione K-Fold.
        Se non viene inserito alcun valore, imposta il valore di default a 10.

        Returns:
            int: Numero di fold da utilizzare per la validazione.
        """
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
        Chiede all'utente di selezionare le metriche da calcolare.
        Se non viene inserito alcun valore, seleziona tutte le metriche di default.

        Returns:
            list: Lista delle metriche selezionate dall'utente.
        """
        lista_metriche = [
        "Accuracy", "Error Rate", "Sensitivity", "Specificity",
        "Geometric Mean", "Tutte le metriche"]

        n = len(lista_metriche)
        numeri_validi = {str(i) for i in range(1, n+1)}  # Numeri validi (da "1" a "7")

        while True:
            print("\nScegliere tra le seguenti metriche quelle che si vogliono calcolare:\n")
            for index, el in enumerate(lista_metriche, start=1):
                print(f"\u25BA {index}. Per selezionare {el} premere {index}\n")

            metriche_scelte = input("Inserisci i numeri delle metriche separati da spazio: ").split()

            # Se l'utente non inserisce nulla, seleziona tutte le metriche
            if not metriche_scelte:
                metriche_scelte = ["6"] 

            # Controlla se tutte le metriche scelte sono valide
            if all(el in numeri_validi for el in metriche_scelte):
                return metriche_scelte  # Se sono tutte valide, restituisce la lista
            else:
                print("\nErrore: Alcuni numeri inseriti non sono validi. Riprova.")

    @staticmethod
    def get_mod_calculation_metrics(num):
        """
        Chiede all'utente la modalità di calcolo delle metriche per gli esperimenti.
        L'utente può scegliere tra il calcolo delle metriche singole o la media delle metriche.

        Args:
            num (int): Numero di esperimenti eseguiti.

        Returns:
            bool: True se l'utente sceglie di calcolare la media delle metriche, False altrimenti.
        """

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
    def want_auc_value():
        scelta = input(f"Si desidera visualizzare il valore di 'Area Under The Curve'.(s/n): ").strip().lower()
        mod=False
        if scelta=="":
            print("Nessuna opzione inserita. Impostato di default la non visualizzazione dell'Area Under The Curve")
            return mod
        
        if scelta == "s":
            mod=True

        return mod
        
        

                    

    @staticmethod
    def get_file():
        """
        Chiede all'utente il nome del file Excel in cui salvare i risultati delle metriche.
        Se non viene inserito alcun nome, usa il valore di default "performance_model.xlsx".

        Returns:
            str: Percorso completo del file in cui salvare i risultati.
        """
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

