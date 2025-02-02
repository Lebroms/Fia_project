
from scripts.data_preprocessing.loader.factory import load_data
from scripts.data_preprocessing.Target_Features.ClassLabel_Selector import classlabel_selector
from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor

import pandas as pd

if __name__ == "__main__":
   
    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
    
    col2=["Irrelevant_Feature1","Irrelevant_Feature2","Sample code number"]
    col3=["Random_String","Irrelevant_Numeric","Sam!"]
    col4=["sample_code_number","randomfeature2","col_11"]
    col5=["irrelevant_col_1","col_0","irrelevant_col_2"]
    col_lab_2_3=["Class"]
    col_lab_4=["class"]
    col_lab_5=["col_10"]

    """queste variabili servono solo per provare il codice con tutti e 5 i file più velocemente,
       nella versione finale queste variabili saranno inserite tramite riga di comando"""

    
    dataset = Df_Processor.elimina_colonne(dataset, col4)
    
    dataset = Df_Processor.crea_dummy_variables(dataset)

    [Features, colonne_label] = classlabel_selector(dataset, col_lab_4)

    Features = Df_Processor.gestisci_valori_mancanti(Features)
    
    Df_Processor.scala_features(Features)

    print(Features.dtypes)
    print(Features)
    print(colonne_label)
   
   
    

