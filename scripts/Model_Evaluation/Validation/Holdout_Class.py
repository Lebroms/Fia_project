import numpy as np

from KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation

from KNN.scegli_k import scegli_k
from ..Metrics.Classe_Metriche import Metriche

from scripts.Model_Evaluation.Metrics.scegli_mod_calcolo_metrics import scegli_metriche,scegli_modalita_calcolo_metriche


class HoldoutValidation(validation):
    """

    Questa classe ha lo scopo di dividere il dataset in due parti: training set e test set.
    Implementa un metodo astratto della classe padre validation.
    
    """

    def __init__(self):

        """
        Costruttore della classe:
        Istanzia un oggetto della classe HoldoutValidation che gestisce il modello di validazione 
        Holdout con attributo "test_size" ovvero la percentuale del dataset da usare come test set inserita
        dall'utente tramite riga di comando 
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

        self.test_size = test_size
        print(f"Impostata la percentuale al {test_size * 100}%")


    def validation(self, features, target):
        """
        metodo importato dalla classe padre validation 

        Serve per dividere il dataset in training e test in base all'attributo della classe test_size.

        In questo metodo viene chiamata la classe esterna Classificatore_Knn per eseguire la predizione
        sul test_set. E viene chiamata anche la classe Metriche per gestire il calcolo delle metriche 

        Parametri:
        features: DataFrame delle features.
        target: DataFrame con le class label.



        """




        num_campioni = len(features) #calcola il numero di campioni (ovvero le righe presenti nel dataframe delle features
        
        indici_test = np.random.choice(num_campioni, size=int(self.test_size * num_campioni), replace=False)
        """
        La funzione random.choice moltiplica il numero di campioni per la percentuale dichiarata
        in ingresso in modo da definire quanti record assegnare a test. Inoltre, impostando il 
        parametro replace a false ogni record può essere selezionato una sola volta. 
        """

        indici_training = list(set(range(num_campioni)) - set(indici_test)) 
        #viene fatta una lista che contiene gli indici dei campioni da usare come training ottenuta sottraendo
        #a tutti gli indici quelli selezionati come test set


        X_training, X_test = features.iloc[indici_training], features.iloc[indici_test]
        #individua in features le righe aventi indici del training e aventi gli indici del test
        #e le assegna ai dataframe X_training e X_test che contengono

        Y_training, Y_test = target.iloc[indici_training], target.iloc[indici_test]
        #individua in target le righe aventi indici del training e aventi gli indici del test
        #e le assegna ai dataframe X_training e X_test che contengono

        
        k=scegli_k() #chiama la funzione scegli_k per scegliere i k vicini
        knn = Classificatore_KNN(X_training, Y_training,k) #crea un istanza di Classificatore
        lista_predizioni = knn.predizione(X_test) 
        #chiama la funzione predizione sull'istanza di classificatore e assegna la predizione delle label del test
        #a una lista
        
        print(lista_predizioni)

        lista_label=Y_test.iloc[:, 0].tolist() 
        #mette i valori del dataframe contenente le label dei campioni
        #usati come test in una lista.
        print(lista_label)




        Metrica= Metriche(lista_label, lista_predizioni)
        #crea un istanza della classe metriche passando
        #le due liste appena assegnate
        Metriche_selezionate = scegli_metriche()#chiama la funzione scegli metriche per selezionare quali calcolare sulla predizione 

        Metriche_Calcolate=Metrica.calcola_metriche(Metriche_selezionate)
        #chiama la funzione calcola_metriche della classe Metriche per calcolare le metriche appena selezionate

        print("Le metriche calcolate sono:")
        for c, v in Metriche_Calcolate.items():
            print(f" \n - {c}: {v:.4f}")







