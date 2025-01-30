import numpy as np
import random
import pandas as pd

# Carichiamo il dataset pre-elaborato
data = pd.read_csv('0simulazione_dataset.csv')

# Supponiamo che features_df e labels_df siano già passati come DataFrame
# features_df contiene le features e labels_df contiene le labels

# Convertiamo i DataFrame in numpy arrays
#features = features_df.values  # Converte le features in un array numpy
#labels = labels_df.values.flatten()  # Converte le labels in un array numpy (appiattito se è una colonna)

# Funzione per calcolare la distanza euclidea tra due punti
def distanza_euclidea(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))

def trova_k_vicini(X_train_set,Y_train_set,X_test,k=3):
    distanze=[]

    for i, X_train_set in X_train_set.iterrows():
        dist=distanza_euclidea(X_train_set,X_test)
        distanze.append((dist,Y_train_set[i]))

        distanze.sort(key=lambda x: x[0]) #ordina la lista delle distanze tra il campione e i vari record in ordine crescente, rispetto alla distanza

        k_vicini=[label for _, label in distanze[:k]] #riporta la lista delle prime k-label, ordinata rispetto alle distanze
        return k_vicini

def predici_label(k_vicini):
    count = {}
    for label in k_vicini:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1

    max_count=max(count.values())
    label_candidate=[label for label,counter in count.items() if counter==max_count]

    return random.choice(label_candidate)





class Classificatore_KNN(self,k):
    
    def __init__(self, k, X_train):
        """
        Inizializza il classificatore K-NN controllando che k sia valido.
        """
        self.n_train = len(X_train)  # Numero di campioni nel training set

        if not isinstance(k, int) or k <= 0:
            print("Errore: k deve essere un intero positivo. Impostato a 3 di default.")
            self.k = 3  # Valore di default
        elif k > self.n_train:
            print(f"Errore: k ({k}) è maggiore del numero di campioni nel training set ({self.n_train}). Impostato a {self.n_train}.")
            self.k = self.n_train  # Impostiamo k uguale al numero di campioni disponibili
        elif k % 2 == 0:
            print(f"Avviso: k ({k}) è pari. Questo potrebbe causare pareggi nella votazione. Considera di usare un valore dispari.")
            self.k = k
        else:
            self.k = k  # Se tutto è corretto, assegniamo k normalmente
    
    def converti_in_array(self,X_train_set,Y_train_set):
        self.X_train_set=np.array(X_train_set)
        self.Y_train_set=np.array(Y_train_set)
        
    def predizione(self,X_test):
        predizione=[]
        for x_test in np.array(X_test):
            k_vicini=trova_k_vicini(self.X_train_set,self.Y_train_set,x_test,self.k)
            label=predici_label(k_vicini)
            predizione.append(label)


