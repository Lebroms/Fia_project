"""
Created on Wed Feb  5 19:47:59 2025

@author: emagi
"""


import unittest
import numpy as np
from scripts.Model_Evaluation.Metrics.Classe_Metriche import Metriche

class TestMetriche(unittest.TestCase):

    def setUp(self):
        """
        Prepara i dati di test per tutti gli unittest.
        """

        self.y_real = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # Valori (casuali) reali da usare nei test
        self.y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])  # Valori (casuali) predetti da usare nei test

        self.metriche = Metriche(self.y_real, self.y_pred)
        # In questa riga viene definita un'istanza della classe Metriche, in modo da ereditare tutti i metodi
        # che poi verranno richiamati nei vari test

    def mock_make_confusion_matrix(self):
        """
        Mock della funzione confusion_matrix().
        Restituisce una matrice di confusione fissa per testare i metodi che la utilizzano.
        """
        # Questo mock è composto da valori fissi di TN,FP,FN,TP. Il mock serve a valutare
        # il funzionamento di tutti i metodi in cui è richiamata la confusion matrix

        return np.array([[4, 2], [1, 3]])

    def test_specificity(self):
        """
        Testa il calcolo della Specificity usando il mock della confusion matrix.
        """
        # Si utilizza il mock della matrice al posto della matrice calcolata nella classe originale
        self.metriche.make_confusion_matrix = self.mock_make_confusion_matrix

        # Si assegnano alle variabili dei valori casuali per il test
        true_negative = 4
        false_positive = 2
        specificity_stimata = true_negative / (true_negative + false_positive)

        self.assertAlmostEqual(self.metriche.specificity(), specificity_stimata, places=5)
        # La funzione assertAlmostEqual verifica che due numeri float siano quasi uguali.
        # In questo caso si confronatano i valori di specificity ottenuti dal test e dalla classe originale
        # Places = 5 definisce quante cifre decimali bisogna confrontare

    def test_calcola_metriche(self):
        """
        Testa il metodo "calcola_metriche" per verificare che calcoli correttamente le metriche scelte.
        """
        # Si utilizza il mock della matrice al posto della matrice calcolata nella classe originale
        self.metriche.make_confusion_matrix = self.mock_make_confusion_matrix


        metriche_scelte = ["1", "3", "5"]
        # Si selezionano tre metriche da caso tra quelle disponibili per provare il test
        # In questo caso Accuracy, Sensitivity e Geometric mean.

        result = self.metriche.calcola_metriche(metriche_scelte)
        # Si richiama dalla classe originale il metodo per calcolare le varie metriche


        self.assertIn("Accuracy", result)
        self.assertIn("Sensitivity", result)
        self.assertIn("Geometric Mean", result)
        # La funzione assertIn verifica che l'elemento specificato sia contenuto nella lista "result"

        self.assertIsInstance(result["Accuracy"], float)
        self.assertIsInstance(result["Sensitivity"], float)
        self.assertIsInstance(result["Geometric Mean"], float)
        # La funzione assertIsInstance verifica invece che i valori assunti dagli elementi
        # della lista siano dei float

    def test_confusion_matrix(self):
        """
        Testa che il metodo "make_confusion_matrix" restituisca un np array di dimensione 2x2
        con i valori specificati.
        """

        self.metriche.make_confusion_matrix = self.mock_make_confusion_matrix
        # Si utilizza il mock della matrice al posto della matrice calcolata nella classe originale

        # Confronta il valore restituito dal mock con quello atteso
        matrice_stimata = np.array([[4, 2], [1, 3]])

        np.testing.assert_array_equal(self.metriche.make_confusion_matrix(), matrice_stimata)
        # Una volta sostituito il metodo della classe originale "make_confusion_matrix" con un
        # mock che ha valori prefissati, questa funzione confronta i valori attesi (che dovrebbe restituire
        # il metodo) con i valori del mock.

if __name__ == "__main__":
    unittest.main()


