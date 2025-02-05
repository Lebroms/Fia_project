# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:53:35 2025

@author: emagi
"""

from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor
import scripts.data_preprocessing.pulizia_dataset.pulizia_data  # Importiamo il modulo dove si trova interfaccia_utente
from mock_standardization import MockInterfaccia_Standardization  # Importiamo il nostro mock
import unittest
import pandas as pd

#  Sovrascriviamo interfaccia_utente dentro il modulo pulizia_data.py con il mock
scripts.data_preprocessing.pulizia_dataset.pulizia_data.interfaccia_utente = MockInterfaccia_Standardization



class TestDfProcessor(unittest.TestCase): 
    """Classe per testare la classe Df_Processor con un mock di interfaccia_utente"""
    
    def setUp(self):

        self.df = pd.DataFrame({
            'Class': [0, 0, 1, 1, 0],
            'Mitosi': [1, 2, 6, 7, 3],
            'Marginal adhesion': [2, 1, 6, 7, 1],
            'Sample code number': [1, 2, 3, 4, 5],
            'Uniformity of Cell Shape': [1, 2, 6, 8, 2],
            'Blood Pressure': [95, 100, 130, 125, 101],
            'Heart Rate': [65, 66, 95, 97, 63],
        })
        
    def test_scalare_features_standardization(self):
         Df_Processor.scala_features(self.df)
         for col in self.df:
             self.assertAlmostEqual(self.df[col].mean(), 0, delta=0.1)
             self.assertAlmostEqual(self.df[col].std(), 1, delta=0.1)
             self.assertTrue(self.df[col].max() <= 3.5 and self.df[col].min() >= -3)
             
             
if __name__ == '__main__':
    unittest.main()             