import numpy as np

from KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation

from KNN.scegli_k import scegli_k



class KfoldValidation(validation):
    """
    Questa classe ha lo scopo di dividere il dataset in K sottogruppi di uguale dimensione, chiamati
    folds. Ad ogni iterazione uno dei folds viene utilizzato come test, mentre i restanti K-1 folds
    rappresentano il training set. Il processo viene ripetuto per tutti i folds.
    """
    def __init__(self):
        """
        Inizializza la suddivisione K-Fold.

        n_folds: Numero di folds in cui suddividere il DataFrame (default: 10).
        shuffle: Parametro per scambiare casualmente l'ordine dei records nel DataFrame. Ovviamente
        i folds devono essere almeno 2 per essere mescolati.
        """
        while True:  # Ciclo per chiedere il valore finché non è valido
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
        Divide il dataset in K-folds senza random_state.

        features: DataFrame delle features.
        target: DataFrame con le class label.
        return: La funzione restituisce in output una lista di tuple (train_indices, test_indices)
        per ogni fold.
       ò """
        num_campioni = len(features)

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
        di campioni e il numero di folds non è un intero il resto viene assegnato ai primi folds.
        """

        k = scegli_k()
        folds = {}
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


            folds[f"fold_test {index+1}"] = (f"Lista predizioni{lista_predizioni}", f"Label test{label_test_set}")

        print(folds)



