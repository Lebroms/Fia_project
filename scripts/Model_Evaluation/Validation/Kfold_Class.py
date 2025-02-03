import numpy as np

from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation

from ...KNN.scegli_k import scegli_k

from ..Metrics.Classe_Metriche import Metriche

from scripts.Model_Evaluation.Metrics.scegli_mod_calcolo_metrics import scegli_metriche,scegli_modalita_calcolo_metriche




class KfoldValidation(validation):
    """

    Questa classe ha lo scopo di dividere il dataset in due parti: training set e test set.
    Implementa un metodo astratto della classe padre validation.

    """
    def __init__(self):
        """
        Costruttore della classe:
        Istanzia un oggetto della classe KfoldValidation che gestisce il modello di validazione
        KfoldValidation con attributo "num_folds" ovvero i sottogruppi di uguale dimensione in cui
        viene diviso il dataset, inserito dall'utente tramite riga di comando.

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

        self.n_folds = n_folds
        print(f"Impostato il numero di fold a {self.n_folds}")



    def validation(self, features, target):
        """
        metodo importato dalla classe padre validation 

        Serve per dividere il dataset in training e test in base all'attributo num_folds facendo tot 
        esperimenti in base a quanti sono i sottogruppi scelti e a ogni iterazione vine usato un fold 
        diverso come test e gli altri num_fold-1 sottogruppi vengono usati come train 

        In questo metodo viene chiamata la classe esterna Classificatore_Knn per eseguire la predizione
        sul test_set. E viene chiamata anche la classe Metriche per gestire il calcolo delle metriche 
        

        Parametri:
        features: DataFrame delle features.
        target: DataFrame con le class label.
        """
        num_campioni = len(features) #calcola il numero di campioni (ovvero le righe presenti nel dataframe delle features)

        indici = np.arange(num_campioni)
        """
        Viene generato un array di indici che va da 0 a num_campioni - 1. Questi indici verranno
        poi mescolati tramite Shuffle e usati per comporre i vari folds
        """


        np.random.shuffle(indici)
        """
        L'ordine degli indici viene scambiato in modo casuale, così da garantire un
        risultato più valido per il test. Questo perchè potrebbero esserci casi in cui i valori 
        di class label sono disposti prima con tutti i 2 e poi con tutti i 4.
        """

        fold_sizes = np.full(self.n_folds, num_campioni // self.n_folds, dtype=int)
        fold_sizes[:num_campioni % self.n_folds] += 1
        """
        Viene generato un array di lunghezza pari al numero di folds, e ogni elemento dell'array
        indica quanti campioni ci sono in ogni fold. Se il risultato del rapporto tra il numero
        di campioni e il numero di folds non è un intero il resto dei campioni viene ridistribuito tra 
        i primi folds.
        """

        k = scegli_k() #chiama la funzione scegli_k per scegliere i k vicini
        folds = {} #crea un dizionario vuoto 

        num=self.n_folds #assegna l'attributo n_folds a num
        

        Metriche_Selezionate =scegli_metriche()

        modalità=scegli_modalita_calcolo_metriche(num)

        lista_metriche=[]#crea un dizionario vuoto

        current = 0
        for index, i in enumerate(fold_sizes):
            indici_test = indici[current:current + i]
            """Seleziona gli indici da 0 a i, che verranno usati come test set."""
            indici_training = np.concatenate((indici[:current], indici[current + i:]))
            """Gli indici non selezionati compongono il training set."""
            feature_test_set = features.iloc[indici_test]
            feature_train_set = features.iloc[indici_training]

            label_test_set = target.iloc[indici_test]
            label_train_set = target.iloc[indici_training]

            #folds.append((indici_training, indici_test))
            """Aggiunge il fold appena costruito alla lista "folds". """
            current += i

            """
            Incrementa l'indice per la prossima iterazione. In questo modo ogni iterazione 
            seleziona un pezzo del dataset come test e il resto come training, e quindi alla
            fine ogni campione verrà utilizzato solo una volta come test.
            """



            knn=Classificatore_KNN(feature_train_set,label_train_set,k)
            lista_predizioni=knn.predizione(feature_test_set)

            lista_label=label_test_set.iloc[:, 0].tolist()


            Metrica= Metriche(lista_label, lista_predizioni)

            Metriche_Calcolate=Metrica.calcola_metriche(Metriche_Selezionate)

            lista_metriche.append((f"Esperimento{i+1}",Metriche_Calcolate))
            

            
            

            
            

            '''if modalità == False:
                print(f"Le metriche selezionate per il test sul fold {index+1} valgono:")
                for c, v in Metriche_Calcolate.items():
                    print(f"  - {c}: {v:.4f}")

                
            else:
                dizionario_metriche[f"Esperimento{i+1}"]=Metriche_Calcolate'''


        

        if modalità: #uscito dal for questo se modalità è True è un dict di dict
            metriche_raccolte = {}

            # Passo 1: Raccogliamo i valori di ogni metrica
            for _, metriche in lista_metriche:
                for nome_metrica, valore in metriche.items():
                    if nome_metrica not in metriche_raccolte:
                        metriche_raccolte[nome_metrica] = []  # Creiamo una lista per raccogliere i valori
                    metriche_raccolte[nome_metrica].append(valore)

            # Passo 2: Calcoliamo la media per ogni metrica
            medie_metriche = {metrica: np.mean(valori) for metrica, valori in metriche_raccolte.items()}
            return [medie_metriche]

        


        else:
            metriche_per_esperimento=[]
            for _,metriche in lista_metriche:
                metriche_per_esperimento.append(metriche)
            return metriche_per_esperimento #fa una lista in cui ogni elemento è un sottodizionario di dizinario_metriche 

            



