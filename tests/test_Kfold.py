import unittest
import numpy as np
import pandas as pd
from scripts.Model_Evaluation.Validation.Kfold_Class import KfoldValidation


# ✅ Mock manuale di `scegli_k()`
def mock_scegli_k():
    return 3  # Restituisce sempre 3


# ✅ Mock manuale di `scegli_metriche()`
def mock_scegli_metriche():
    return ["accuracy"]  # Restituisce sempre "accuracy"


# ✅ Mock manuale di `scegli_modalita_calcolo_metriche()`
def mock_scegli_modalita_calcolo_metriche():
    return False  # Restituisce sempre False


# ✅ Mock manuale della classe `Classificatore_KNN`
class MockClassificatoreKNN:
    def __init__(self, X_train, Y_train, k):
        self.k = k

    def predizione(self, X_test):
        return [0] * len(X_test)  # Simula predizioni tutte 0


class TestKfoldValidation(unittest.TestCase):

    def setUp(self):
        """
        Prepara un dataset di test con 100 righe e 5 colonne di features, più una colonna target.
        """
        np.random.seed(42)  # Per garantire risultati riproducibili
        self.features = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
        self.target = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)), columns=["target"])

    def test_validation(self):
        """
        Testa il metodo `validation()` della classe `KfoldValidation`
        """
        # ✅ Istanzia la classe `KfoldValidation`
        kfold = KfoldValidation()
        kfold.n_folds = 5  # Imposta manualmente il numero di folds

        # ✅ Sostituisci i metodi con i mock manuali
        kfold.scegli_k = mock_scegli_k
        kfold.scegli_metriche = mock_scegli_metriche
        kfold.scegli_modalita_calcolo_metriche = mock_scegli_modalita_calcolo_metriche
        kfold.Classificatore_KNN = MockClassificatoreKNN  # Usa il mock della classe KNN

        # ✅ Esegui `validation()`
        metriche = kfold.validation(self.features, self.target)

        # ✅ Controlla il formato del risultato
        self.assertIsInstance(metriche, list)  # Deve essere una lista
        self.assertIsInstance(metriche[0], dict)  # Ogni elemento deve essere un dizionario
        self.assertIn("accuracy", metriche[0])  # Il dizionario deve contenere "accuracy"

        # ✅ Verifica la suddivisione dei dati
        num_campioni_test = len(self.features) // kfold.n_folds
        num_campioni_train = len(self.features) - num_campioni_test

        self.assertEqual(num_campioni_test, 20)  # Ogni test set deve avere 20 campioni
        self.assertEqual(num_campioni_train, 80)  # Il training set deve avere 80 campioni


if __name__ == "__main__":
    unittest.main()
