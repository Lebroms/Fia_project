import unittest
import pandas as pd
from scripts.Model_Evaluation.Validation.Kfold_Class import KfoldValidation


"""
Vengono innanzitutto definiti i mock, ovvero elementi di prova che verranno utilizzati
nel test per valutare il funzionamento dei vari codici inerenti alla classe KfoldValidation. 
"""

def mock_scegli_k():
    return 3  # Per il test si imposta il valore di default k=3, nel caso in cui non venga specificato


def mock_scegli_metriche():
    return ["Tutte le metriche"]  # Codice della metrica che si intende valutare


def mock_scegli_modalita_calcolo_metriche():
    return False  # Si sceglie se si vuole visualizzare la media delle metriche di tutti i folds oppure i singoli valori


class MockClassificatoreKNN:
    def __init__(self, X_train, Y_train, k):
        self.k = k

    def predizione(self, X_test):
        return [0] * len(X_test)
    """
    Nel test impostiamo tutti i valori delle predizioni a 0 perchè si vuole solamente
    valutare la logica del codice. Impostare risultati variabili renderebbe il test inutilmente più complicato.
    """


class TestKfoldValidation(unittest.TestCase):

    def setUp(self):

        Features = {
            "features_1": [1, 3, 6, 9, 5, 6, 5, 7, 6, 4],
            "features_2": [6, 4, 5, 3, 2, 1, 9, 5, 4, 1],
            "features_3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 2],
            "features_4": [2, 9, 4, 1, 7, 8, 3, 1, 6, 9],
            "features_5": [2, 8, 2, 5, 2, 0, 2, 2, 6, 1],
            "features_6": [1, 3, 6, 9, 5, 6, 5, 7, 6, 4],
            "features_7": [6, 4, 5, 3, 2, 1, 9, 5, 4, 1],
            "features_8": [1, 3, 6, 9, 5, 6, 5, 7, 6, 4],
            "features_9": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        }

        self.dfFeatures = pd.DataFrame(Features)

        Target = {

            "target": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
        }

        self.dfTarget = pd.DataFrame(Target)

    """
    Viene definito un dataset di prova composto da 9 colonne features (features_1, features_2, ecc.) e 
    una target, con valori casuali.   
    """

    def test_validation(self):

        kfold = KfoldValidation()

        kfold.scegli_k = mock_scegli_k
        kfold.scegli_metriche = mock_scegli_metriche
        kfold.scegli_modalita_calcolo_metriche = mock_scegli_modalita_calcolo_metriche
        kfold.Classificatore_KNN = MockClassificatoreKNN

        """
        Viene richiamata la classe KfoldValidation, e vengono assegnati a kfold i mock definiti precedentemente.
        """

        metriche = kfold.validation(self.dfFeatures, self.dfTarget)

        self.assertIsInstance(metriche, list)
        self.assertIsInstance(metriche[0], dict)
        for metrica in ["Accuracy", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean"]:
            self.assertIn(metrica, metriche[0])

        """
        Il metodo assertIsInstance serve a verificare che l'oggetto che si trova come primo parametro
        sia effettivamente del tipo specificato nel secondo parametro.
        Mentre assertIn verifica che l'elemento specificato come primo parametro sia contenuto
        nell'oggetto specificato nel secondo parametro.
        """

if __name__ == "__main__":
    unittest.main()

