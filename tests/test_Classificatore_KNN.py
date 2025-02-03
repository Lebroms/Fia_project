# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:51:29 2025

@author: emagi
"""

import unittest
import numpy as np
import pandas as pd
from scripts.KNN.Classificatore_Knn import Classificatore_KNN, trova_k_vicini, predici_label

class TestClassificatoreKNN(unittest.TestCase):
    """
    """
   
    def setUp(self):

        # dataset di esempio
        self.X_train = pd.DataFrame({
            'Feature1': [1, 5, 10],
            'Feature2': [1, 5, 10]
        })

        self.Y_train = pd.DataFrame({
            'Label': [0, 0, 1]
        })

        self.X_test = pd.DataFrame({
            'Feature1': [6],
            'Feature2': [6]
        })

        self.k = 2  # Numero di vicini da considerare

        # Creiamo un'istanza del classificatore
        self.knn = Classificatore_KNN(self.X_train, self.Y_train, self.k)

    def test_trova_k_vicini(self):
        """
        Test sulla funzione trova_k_vicini per verificare che restituisca i vicini corretti.
        """
        k_vicini = trova_k_vicini(self.X_train, self.Y_train, np.array([5, 6]), self.k)
       
        self.assertEqual(len(k_vicini), 2)  # Dovrebbero essere esattamente k=2 vicini ovvero (1,1) e (5,5)

    def test_predici_label(self):
        """
        Test sulla funzione predici_label per verificare che restituisca il valore più frequente tra i vicini.
        """
        k_vicini = [0, 1, 1]  # Il valore più frequente è 1
        label_predetta = predici_label(k_vicini)
        self.assertEqual(label_predetta, 1)

    def test_predizione(self):
        """
        Test sul metodo predizione() della classe Classificatore_KNN.
        """
        prediction = self.knn.predizione(self.X_test)
        self.assertIsInstance(prediction, list)  # La predizione deve restituire una lista
        self.assertEqual(len(prediction), len(self.X_test))  # Il numero di predizioni deve corrispondere al numero di dati di test
        self.assertIn(prediction[0], [0.0, 1.0])  # La classe prevista deve essere 0 o 1

if __name__ == '__main__':
    unittest.main()
