import pandas as pd

import random 
from KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation
from KNN.scegli_k import scegli_k



class RandomSubsamplingValidation(validation):
    def __init__(self):

        while True:  # Ciclo per chiedere il valore finché non è valido
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

        self.num_experiments = int(num_experiments)
        print(f"Impostato il numero di esperimenti a {self.num_experiments}")

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
        
        k=scegli_k()
        risultati_esperimenti={}

        for i in range(self.num_experiments):  # Ripetiamo per ogni esperimento
            n = len(features)  # Numero totale di campioni
            n_campioni_test = int(n * self.test_size)  # Numero di campioni per il test set

            #Crea una lista di indici e li mescoliamo
            indici = list(range(n))  # Lista degli indici [0, 1, 2, ..., n-1]
            random.shuffle(indici)  # Mischiamo gli indici casualmente

            # Selezioniamo indici per train e test
            indici_test= indici[:n_campioni_test]  # scegliamo i primi indici come indici dei campioni da usare come test  `test_size` indici come test set
            indici_train = indici[n_campioni_test:]  # Il resto sono gli indici dei campioni del training set

            # Creiamo i dataset train e test usando gli indici selezionati
            X_train = features.iloc[indici_train]
            y_train = target.iloc[indici_train]
            X_test = features.iloc[indici_test]
            y_test = target.iloc[indici_test]

            knn = Classificatore_KNN(X_train, y_train,k)
            lista_predizioni = knn.predizione(X_test)


            risultati_esperimenti[f"Esperimento {i+1}"]=lista_predizioni

        print(risultati_esperimenti)




