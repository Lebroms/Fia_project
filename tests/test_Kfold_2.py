# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:18:14 2025

@author: emagi
"""

import unittest
import numpy as np
import pandas as pd
from scripts.Model_Evaluation.Validation.Kfold_Class import KfoldValidation


#  Mock per Classificatore_KNN
class MockClassificatoreKNN:
    def __init__(self, X_train, Y_train, k):
        pass

    def predizione_max(self, X_test):
        """
        Simula la predizione:
        - La prima lista è la predizione delle label (0 e 1).
        - La seconda lista è la percentuale di positivi tra i k vicini.
        """
        num_samples = len(X_test)
        return [0] * num_samples, [50] * num_samples  # Predizioni fisse e percentuale di positivi sempre 50%

    def predict_label_by_threshold(self, lista_perc_of_pos):
        """
        Simula il comportamento di soglie diverse:
        - Se la percentuale è sopra la soglia, predice 1, altrimenti 0.
        """
        dict_di_liste = {}
        for threshold in range(0, 101, 10):  # Simuliamo soglie da 0% a 100%
            dict_di_liste[f"threshold:{threshold}%"] = [1 if perc >= threshold else 0 for perc in lista_perc_of_pos]
        return dict_di_liste


class MockMetriche:
    def __init__(self, y_true, y_pred):
        pass

    def make_confusion_matrix(self):
        """Simula una matrice di confusione 2x2 con valori fittizi."""
        return np.array([[5, 2], [3, 4]])  # TN, FP, FN, TP

    def calcola_metriche(self, metriche_selezionate):
        """Restituisce metriche fittizie per i test."""
        return {metrica: 0.8 for metrica in metriche_selezionate}  # Valori fissi

    def costruzione_punti_roc_curve(self, dict_predizioni_con_threshold):
        """Genera punti fittizi per la ROC curve."""
        return [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]  # Punti ROC fittizi


class TestKfoldValidation(unittest.TestCase):
    """Test per la classe KfoldValidation"""

    def setUp(self):
        """Inizializzazione del test con dataset di esempio"""
        self.n_folds = 3
        self.k = 5
        self.modalità = True
        self.metriche_selezionate = ["1", "2", "3"]

        # piccolo dataset di esempio (9 righe)
        self.features = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'Feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2]
        })
        self.target = pd.DataFrame({'Label': [0, 1, 0, 1, 0, 1, 0, 1, 0]})

        # Sostituiamo i veri Classificatore_KNN e Metriche con i mock
        self.Kfold = KfoldValidation(self.n_folds, self.k, self.modalità)
        self.Kfold.Classificatore_KNN = MockClassificatoreKNN  # Sostituiamo il classificatore
        self.Kfold.Metriche = MockMetriche  # Sostituiamo le metriche

    def test_validation(self):
        """Testa il metodo validation() per la corretta esecuzione"""
        metriche_risultato, liste_matrix, liste_di_punti = self.Kfold.validation(self.features, self.target, self.metriche_selezionate)

        #  Verifica il numero di metriche restituite
        self.assertEqual(len(metriche_risultato), 1)  # Deve esserci un dizionario se modalità=True
        self.assertEqual(len(liste_matrix), self.n_folds)  # Devono esserci n_folds matrici di confusione
        self.assertEqual(len(liste_di_punti), self.n_folds)  # Devono esserci n_folds liste di punti ROC

        #  Controlliamo il formato delle confusion matrix
        for matrice in liste_matrix:
            self.assertEqual(matrice.shape, (2, 2))  # Deve essere una matrice 2x2

        #  Controlliamo che le liste di punti ROC abbiano il formato corretto
        for lista_punti in liste_di_punti:
            self.assertTrue(all(len(p) == 2 for p in lista_punti))  # Ogni punto deve avere due coordinate (x, y)

if __name__ == '__main__':
    unittest.main()
