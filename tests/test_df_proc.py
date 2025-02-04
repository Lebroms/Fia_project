# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:26:54 2025

@author: emagi
"""

from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor
import scripts.data_preprocessing.pulizia_dataset.pulizia_data  # Importiamo il modulo dove si trova interfaccia_utente
from mock_interfaccia_utente import MockInterfacciaUtente  # Importiamo il nostro mock
import unittest
import pandas as pd

#  Sovrascriviamo interfaccia_utente dentro il modulo pulizia_data.py con il mock
scripts.data_preprocessing.pulizia_dataset.pulizia_data.interfaccia_utente = MockInterfacciaUtente



class TestDfProcessor(unittest.TestCase): 
    """Classe per testare la classe Df_Processor con un mock di interfaccia_utente"""
    
    def setUp(self):
        self.df = pd.DataFrame({
            'Class': ["benigno", "benigno", "maligno", "maligno", "benigno"],
            'Mitosi': [1, 2, 6, 7, None],
            'Marginal adhesion': [None, 1, 6, 7, 1],
            'Sample code number': [1, 2, 3, 4, 5],
            'Uniformity of Cell Shape': [1, 2, 6, 8, 2],
            'Blood Pressure': [95, 100, 130, 125, 101],
            'Heart Rate': [65, 66, 95, 97, 63],
        })

        self.df_2 = pd.DataFrame({
            'Class': [0, 0, 1, 1, 0],
            'Mitosi': [1, 2, 6, 7, 3],
            'Marginal adhesion': [2, 1, 6, 7, 1],
            'Sample code number': [1, 2, 3, 4, 5],
            'Uniformity of Cell Shape': [1, 2, 6, 8, 2],
            'Blood Pressure': [95, 100, 130, 125, 101],
            'Heart Rate': [65, 66, 95, 97, 63],
        })

    def test_elimina_colonne(self):
        df_after_drop = Df_Processor.elimina_colonne(self.df)
        self.assertNotIn("Blood Pressure", df_after_drop.columns)
        self.assertNotIn("Sample code number", df_after_drop.columns)
        self.assertNotIn("Heart Rate", df_after_drop.columns)

    def test_crea_dummy_variables(self):
        df_with_dummies = Df_Processor.crea_dummy_variables(self.df)
        self.assertIn("Class", df_with_dummies.columns)
        self.assertTrue(df_with_dummies['Class'].dtype == 'int64')
        self.assertEqual(df_with_dummies['Class'].sum(), 2)

    def test_gestisci_valori_mancanti(self):
        df_filled = Df_Processor.gestisci_valori_mancanti(self.df)
        self.assertNotEqual(df_filled['Mitosi'].iloc[4], None)
        self.assertNotEqual(df_filled['Marginal adhesion'].iloc[0], None)

    def test_scalare_features_normalization(self):
        Df_Processor.scala_features(self.df_2)
        for col in self.df_2:
            self.assertTrue(self.df_2[col].max() <= 1 and self.df_2[col].min() >= 0)


if __name__ == '__main__':
    unittest.main()
