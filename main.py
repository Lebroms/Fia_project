
from scripts.data_preprocessing.loader.factory import load_data
from scripts.data_preprocessing.Target_Features.ClassLabel_Selector import Class_label_selector
from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor

import pandas as pd

from scripts.Model_Evaluation.Validation.validation_factory import validation_factory

from scripts.Model_Evaluation.Metrics.visualizzazione_performance import salva_metriche_su_excel
from scripts.interfaccia_utente import interfaccia_utente

if __name__ == "__main__":

    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
    
    columns_to_drop=interfaccia_utente.get_columns_to_drop_input(dataset)  
    dataset = Df_Processor.elimina_colonne(dataset, columns_to_drop) # elimina le colonne che non si desiderano
    
    dataset = Df_Processor.crea_dummy_variables(dataset) # converte le colonne che sono del tipo string in valori numerici usando le dummy variables

    colonne_target = interfaccia_utente.get_target_columns()
    Features, colonne_label = Class_label_selector.select_label(dataset, colonne_target) # divide il dataframe in due sotto dataframe: feature e label

    strategy=interfaccia_utente.get_replacement_stretegy()   
    Features = Df_Processor.gestisci_valori_mancanti(Features,strategy) # gestisce i valori nan nelle features
    
    metodo=interfaccia_utente.get_scaling_method()    
    Df_Processor.scala_features(Features, metodo) # scala i valori nelle features

    print(Features.dtypes)
    print(Features)
    print(colonne_label)


    #-------------------------------parte per testare Holdout
    
    strategy=interfaccia_utente.get_validation_method()
    validators=validation_factory.get_validation_class(strategy)
    lista_metriche=validators.validation(Features,colonne_label)


    salva_metriche_su_excel(lista_metriche)



    


   
   
    


