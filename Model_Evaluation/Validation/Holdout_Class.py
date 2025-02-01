import numpy as np

class HoldoutValidation:
    """
    Questa classe ha lo scopo di dividere il dataset in due parti: training e test. In input la classe
    riceve i due dataframe, corrispondenti alle features e a class label. Inoltre, la classe riceve
    il valore percentuale da assegnare al test, e al training di conseguenza.
    """

    def __init__(self, test_size=0.2, random_state=None):
        """

        test_size: Percentuale di dataset da usare per il test (default: 20%).
        random_state: Indica se mantenere fissi i record assegnati a test e training.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split(self, features, target):
        """
        Divide il dataset in training e test.

        features: DataFrame delle features.
        target: DataFrame con le class label.
        """

        if self.random_state:
            np.random.seed(self.random_state)
            """
            Se random_state assume il valore 42, i record assegnati a test e training
            rimangono gli stessi ad ogni iterazione, in caso contrario
            test e training assumono ogni volta un nuovo set di record.
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

        return X_training, X_test, Y_training, Y_test



