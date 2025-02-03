
from scripts.data_preprocessing.loader.factory import load_data
from scripts.data_preprocessing.Target_Features.ClassLabel_Selector import classlabel_selector
from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor

import pandas as pd

from scripts.Model_Evaluation.Validation.validation_factory import validation_factory

from scripts.Model_Evaluation.Metrics.visualizzazione_performance import salva_metriche_su_excel


if __name__ == "__main__":
    
    

    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
   
    dataset = Df_Processor.elimina_colonne(dataset) # elimina le colonne che non si desiderano
    
    dataset = Df_Processor.crea_dummy_variables(dataset) # converte le colonne che sono del tipo string in valori numerici usando le dummy variables

    Features, colonne_label = classlabel_selector(dataset) # divide il dataframe in due sotto dataframe: feature e label

    Features = Df_Processor.gestisci_valori_mancanti(Features) # gestisce i valori nan nelle features
    
    Df_Processor.scala_features(Features) # scala i valori nelle features

    print(Features.dtypes)
    print(Features)
    print(colonne_label)


    #-------------------------------parte per testare Holdout


    validators=validation_factory.getvalidationstrategy()
    lista_metriche=validators.validation(Features,colonne_label)


    salva_metriche_su_excel(lista_metriche)



    


   
   
    


