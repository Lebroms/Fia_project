# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:53:35 2025

@author: emagi
"""

from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor
from mock_standardization import MockInterfaccia_Standardization  
import unittest
import pandas as pd

class TestDf_Standardization(unittest.TestCase): 
    """Classe per testare il metodo scala_features della classe Df_Processor
        quando l'utente sceglie il metodo standardization"""
        
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
        
    def test_scala_features_standardization(self):
        """
        Verifica che il metodo applichi la standardizzazione per ogni colonna del df
        """
        metodo=MockInterfaccia_Standardization.get_scaling_method()
        Df_Processor.scala_features(self.df, metodo)
        for col in self.df:
             self.assertAlmostEqual(self.df[col].mean(), 0, delta=0.1) # verifica che i valori di ogni colonna hanno media nulla
             self.assertAlmostEqual(self.df[col].std(), 1, delta=0.1)  # verifica che i valori di ogni colonna hanno std=1
             self.assertTrue(self.df[col].max() <= 3.5 and self.df[col].min() >= -3)  
             # verifica che i valori di ogni colonna sono compresi tra -3 e 3 ma con un possibile delta=0.5 dovuto agli outlier
             
if __name__ == '__main__':
    unittest.main()             