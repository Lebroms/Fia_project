# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:50:28 2025

@author: emagi
"""


from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor

import unittest
import pandas as pd



class TestDfProcessor(unittest.TestCase): 
    """
    Classe per verificare il corretto funzionamento di tutti i metodi statici della classe Df_Processor.
    La classe eredita tutti gli attributi e i metodi, per effettuare i test, della classe unittest.TestCase     
    """
    def setUp(self):
        #  DataFrame di esempio per i test
        self.df = pd.DataFrame({
            'Class': ["benigno","benigno","maligno","maligno","benigno"],
            'Mitosi': [1,2,6,7,None],
            'Marginal adhesion': [None,1,6,7,1],
            'Sample code number': [1, 2, 3, 4, 5],
            'Uniformity of Cell Shape': [1,2,6,8,2],
            'Blood Pressure': [95,100,130,125,101],
            'Heart Rate': [65,66,95,97,63],
        })
        # df per verificare la crezionie di dummy anche con più di una colonna di tipo stringa e con più di due valori
        self.dummy_in_piu=pd.DataFrame({
            'Colore': ["rosso","verde","blu","rosso","rosso"]
            })
        self.df_2 = pd.DataFrame({   #df aggiuntivo per testare lo scaling senza aver valori stringa o nan
            'Class': [0,0,1,1,0],    # e senza dover chiamare crea_dummy_variables e gestisci_valori_mancanti
            'Mitosi': [1,2,6,7,3],
            'Marginal adhesion': [2,1,6,7,1],
            'Sample code number': [1, 2, 3, 4, 5],
            'Uniformity of Cell Shape': [1,2,6,8,2],
            'Blood Pressure': [95,100,130,125,101],
            'Heart Rate': [65,66,95,97,63],
        })

    def test_elimina_colonne(self):
        
        df_after_drop = Df_Processor.elimina_colonne(self.df)
        self.assertNotIn("Blood Pressure", df_after_drop.columns) # controlla che le colonne rimosse non sono più nel df
        self.assertNotIn("Sample code number", df_after_drop.columns) # con colonne di defult
        self.assertNotIn("Heart Rate", df_after_drop.columns)

    def test_crea_dummy_variables(self):
        
        df_with_dummies = Df_Processor.crea_dummy_variables(self.df)
        self.assertIn("Class", df_with_dummies.columns)     # controlla che Class sia nel nuovo df con le dummies
        self.assertTrue(df_with_dummies['Class'].dtype == 'int64')    # verifica il corretto comportamento del parametro dtype di pd.get_dummies
        self.assertEqual(df_with_dummies['Class'].sum(), 2)   # benigno=0 maligno=1, somma i valori sulla colonna class
        
        df_piu_dummies=pd.concat([self.df, self.dummy_in_piu], axis=1)  # crea un df con più colonne stringa all'interno più di 2 valori
        df_with_dummies_2 = Df_Processor.crea_dummy_variables(df_piu_dummies)
        self.assertIn("Class", df_with_dummies_2.columns) # controlla che le colonne stringa con solo due valori diventano una dummy con lo stesso nome
        self.assertIn("rosso", df_with_dummies_2.columns) # se più di un nome le chiama con il valore
        self.assertNotIn("blu", df_with_dummies_2.columns) # verifica drop_first=True
        

    def test_gestisci_valori_mancanti(self):
        
        df_filled = Df_Processor.gestisci_valori_mancanti(self.df)
        self.assertNotEqual(df_filled['Mitosi'].iloc[4], None)  # Verifica che il NaN sia stato sostituito
        self.assertNotEqual(df_filled['Marginal adhesion'].iloc[0], None)
        
    def test_scalare_features_normalization(self):

        Df_Processor.scala_features(self.df_2)  # verifica che i valori di ogni colonna sono compresi tra 0 e 1
        for col in self.df_2:
            self.assertTrue(self.df_2[col].max() <= 1 and self.df_2[col].min() >= 0)

    def test_scalare_features_standardization(self):
        
        Df_Processor.scala_features(self.df_2)
        for col in self.df_2:
            self.assertAlmostEqual(self.df_2[col].mean(), 0, delta=0.1) # verifica che i valori di ogni colonna hanno media nulla
            self.assertAlmostEqual(self.df_2[col].std(), 1, delta=0.1)  # verifica che i valori di ogni colonna hanno std=1
            self.assertTrue(self.df_2[col].max() <= 3.5 and self.df_2[col].min() >= -3)
            # verifica che i valori di ogni colonna sono compresi tra -3 e 3 ma con un possibile delta=0.5 dovuto agli outlier

if __name__ == '__main__':
    unittest.main()