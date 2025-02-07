"""
Created on Tue Feb  4 18:47:23 2025

@author: emagi
"""
class MockInterfacciaUtente:
    """Mock della classe interfaccia_utente per evitare input da console nei test"""
    
    @staticmethod
    def get_columns_to_drop_input(df):
        return ["Blood Pressure", "Sample code number", "Heart Rate"]  # Simuliamo l'input dell'utente

    @staticmethod
    def get_replacement_stretegy():
        return "media"  # Simuliamo che l'utente scelga la media

    @staticmethod
    def get_scaling_method():
        return "normalization"  



