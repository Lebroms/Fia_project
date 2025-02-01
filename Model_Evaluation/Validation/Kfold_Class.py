import numpy as np

class KfoldValidation:
    """
    Questa classe ha lo scopo di dividere il dataset in K sottogruppi di uguale dimensione, chiamati
    folds. Ad ogni iterazione uno dei folds viene utilizzato come test, mentre i restanti K-1 folds
    rappresentano il training set. Il processo viene ripetuto per tutti i folds.
    """
    def __init__(self, n_folds=10, shuffle=True):
        """
        Inizializza la suddivisione K-Fold.

        n_folds: Numero di folds in cui suddividere il DataFrame (default: 10).
        shuffle: Parametro per scambiare casualmente l'ordine dei records nel DataFrame. Ovviamente
        i folds devono essere almeno 2 per essere mescolati.
        """
        if n_folds < 2:
            raise ValueError("Il numero di folds deve essere almeno 2")
        self.n_folds = n_folds
        self.shuffle = shuffle

    def split(self, features, target):
        """
        Divide il dataset in K-folds senza random_state.

        features: DataFrame delle features.
        target: DataFrame con le class label.
        return: La funzione restituisce in output una lista di tuple (train_indices, test_indices)
        per ogni fold.
        """
        num_campioni = len(features)

        indici = np.arange(num_campioni)
        """
        Viene generato un array di indici che va da 0 a num_campioni - 1. Questi indici verranno
        poi mescolati tramite Shuffle e usati per comporre i vari folds
        """

        if self.shuffle:
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
        di campioni e il numero di folds non è un intero il resto viene assegnato al primo fold.
        """

        folds = []
        current = 0
        for i in fold_sizes:
            indici_test = indici[current:current + i]
            """Seleziona gli indici da 0 a i, che verranno usati come test set."""
            indici_training = np.concatenate((indici[:current], indici[current + i:]))
            """Gli indici non selezionati compongono il training set."""
            folds.append((indici_training, indici_test))
            """Aggiunge il fold appena costruito alla lista "folds". """
            current += i
            """
            Incrementa l'indice per la prossima iterazione. In questo modo ogni iterazione 
            seleziona un pezzo del dataset come test e il resto come training, e quindi alla
            fine ogni campione verrà utilizzato solo una volta come test.
            """

        return folds
