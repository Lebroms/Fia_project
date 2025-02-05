# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:02:52 2025

@author: emagi
"""

class MockInterfaccia_Standardization:
    """Mock della classe interfaccia_utente per evitare input da console nei test"""
    
    @staticmethod
    def get_scaling_method():
        return "standardization"  # Simula il comportamento della normalizzazione