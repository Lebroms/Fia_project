import numpy as np
import random
import pandas as pd


# Carichiamo il dataset pre-elaborato


# Supponiamo che features_df e labels_df siano già passati come DataFrame
# features_df contiene le features e labels_df contiene le labels

# Convertiamo i DataFrame in numpy arrays
# features = features_df.values  # Converte le features in un array numpy
# labels = labels_df.values.flatten()  # Converte le labels in un array numpy (appiattito se è una colonna)

# Funzione per calcolare la distanza euclidea tra due punti
def distanza_euclidea(x1, x2):
    return np.sqrt(sum((x1 - x2) ** 2))


def trova_k_vicini(X_train_set, Y_train_set, X_test, k=3):
    distanze = []

    for idx, row in X_train_set.iterrows():
        dist = distanza_euclidea(row.values, X_test)
        distanze.append((dist, Y_train_set.loc[idx]))

        distanze.sort(key=lambda x: x[0])  # ordina la lista delle distanze tra il campione e i vari record in ordine crescente, rispetto alla distanza

        k_vicini = [label for _, label in distanze[:k]]  # riporta la lista delle prime k-label, ordinata rispetto alle distanze

    return k_vicini


def predici_label(k_vicini):
    count = {}
    for label in k_vicini:
        label = label.item()
        if label in count:
            count[label] += 1
        else:
            count[label] = 1

    max_count = max(count.values())
    label_candidate = [label for label, counter in count.items() if counter == max_count]

    return random.choice(label_candidate)





class Classificatore_KNN:
    def __init__(self, X_train, Y_train,k):

        self.k=k
        self.X_train_set = X_train
        self.Y_train_set = Y_train

    def predizione(self, X_test):
        predizione = []
        for x_test in np.array(X_test):
            k_vicini = trova_k_vicini(self.X_train_set, self.Y_train_set, x_test, self.k)
            label = predici_label(k_vicini)
            predizione.append(label)

        return [float(x) for x in predizione]



