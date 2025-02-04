import numpy as np
import random
import pandas as pd







class Classificatore_KNN:
    def __init__(self, X_train, Y_train,k):

        self.k=k
        self.X_train_set = X_train
        self.Y_train_set = Y_train

    
    def __distanza_euclidea(self,x1, x2):
        
        return np.sqrt(sum((x1 - x2) ** 2))


    def __trova_k_vicini(self,X_train_set, Y_train_set, X_test, k=3):
        distanze = []

        for idx, row in X_train_set.iterrows():
            
            dist = self.__distanza_euclidea(row.values, X_test)
            
            distanze.append((dist, int(Y_train_set.loc[idx].values)))
            

        distanze.sort(key=lambda x: x[0])  # ordina la lista delle distanze tra il campione e i vari record in ordine crescente, rispetto alla distanza

        k_vicini = [label for _, label in distanze[:k]]  # riporta la lista delle prime k-label, ordinata rispetto alle distanze
            

        return k_vicini


    def __predici_label(self,k_vicini):
        count = {}
        
        for label in k_vicini:
            
            
            if label in count:
                count[label] += 1
            else:
                count[label] = 1

        max_count = max(count.values())
        label_candidate = [label for label, counter in count.items() if counter == max_count]

        return random.choice(label_candidate)
        

    def predizione(self, X_test):
        predizione = []
        for x_test in np.array(X_test):
            
            
            k_vicini = self.__trova_k_vicini(self.X_train_set, self.Y_train_set, x_test, self.k)
            label = self.__predici_label(k_vicini)
            predizione.append(label)

        return [float(x) for x in predizione]



