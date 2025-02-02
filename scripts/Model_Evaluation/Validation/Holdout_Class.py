import numpy as np

from KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation

from KNN.scegli_k import scegli_k
from ..Metrics.Classe_Metriche import Metriche

from scripts.Model_Evaluation.Metrics.scegli_mod_calcolo_metrics import scegli_metriche,scegli_modalita_calcolo_metriche


class HoldoutValidation(validation):
    """
    Questa classe ha lo scopo di dividere il dataset in due parti: training e test. In input la classe
    riceve i due dataframe, corrispondenti alle features e a class label. Inoltre, la classe riceve
    il valore percentuale da assegnare al test, e al training di conseguenza.
    """

    def __init__(self):
        """

        test_size: Percentuale di dataset da usare per il test (default: 20%).

        """

        while True:  # Ciclo per chiedere il valore finché non è valido
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
        Divide il dataset in training e test.

        features: DataFrame delle features.
        target: DataFrame con le class label.
        """




        num_campioni = len(features)
        indici_test = np.random.choice(num_campioni, size=int(self.test_size * num_campioni), replace=False)
        """
        La funzione random.choice moltiplica il numero di features per la percentuale dichiarata
        in ingresso in modo da definire quanti record assegnare a test. Inoltre, impostando il 
        parametro replace a false ogni record può essere selezionato una sola volta. 
        """
        indici_training = list(set(range(num_campioni)) - set(indici_test))

        X_training, X_test = features.iloc[indici_training], features.iloc[indici_test]
        Y_training, Y_test = target.iloc[indici_training], target.iloc[indici_test]
        """
        Vengono assegnati ad X_training e X_test i record delle features e 
        ad y_training e y_test i record della class label. La funzione .iloc viene utilizzata per
        selezionare righe o colonne in un DataFrame in base agli indici numerici.
        """
        k=scegli_k()
        knn = Classificatore_KNN(X_training, Y_training,k)
        lista_predizioni = knn.predizione(X_test)
        print(lista_predizioni)
        lista_label=Y_test.iloc[:, 0].tolist()
        print(lista_label)




        
        

        

            





        Metrica= Metriche(lista_label, lista_predizioni)
        Metriche_selezionate = scegli_metriche()
        Metriche_Calcolate=Metrica.calcola_metriche(Metriche_selezionate)

        print("Le metriche calcolate sono:")
        for c, v in Metriche_Calcolate.items():
            print(f" \n - {c}: {v:.4f}")







