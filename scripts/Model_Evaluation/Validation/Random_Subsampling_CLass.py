import pandas as pd

import random 
# from .KNN import Classificatore_KNN

from classe_validation import validation

class RandomSubsamplingValidation(validation):
    def __init__(self, num_experiments=10, dim_test_set=0.3):
        self.num_experiments = num_experiments  # Numero di esperimenti da eseguire
        self.dim_test_set = dim_test_set  # Percentuale di dati nel test set

    def evaluate(self, X, y):
        

        accuracies = []  # Lista per salvare l’accuratezza di ogni esperimento

        for _ in range(self.num_experiments):  # Ripetiamo per ogni esperimento
            n = len(X)  # Numero totale di campioni
            n_campioni_test = int(n * self.dim_test_set)  # Numero di campioni per il test set

            #Crea una lista di indici e li mescoliamo
            indici = list(range(n))  # Lista degli indici [0, 1, 2, ..., n-1]
            random.shuffle(indici)  # Mischiamo gli indici casualmente

            # 2️⃣ Selezioniamo indici per train e test
            indici_test= indici[:n_campioni_test]  # scegliamo i primi indici come indici dei campioni da usare come test  `test_size` indici come test set
            indici_train = indici[n_campioni_test:]  # Il resto sono gli indici dei campioni del training set

            # 3️⃣ Creiamo i dataset train e test usando gli indici selezionati
            X_train = [X[i] for i in indici_train]
            y_train = [y[i] for i in indici_train]
            X_test = [X[i] for i in indici_test]
            y_test = [y[i] for i in indici_test]

            #Mancano le istruzioni in cui si chiama il knn

        return X_train,y_train,X_test,y_test