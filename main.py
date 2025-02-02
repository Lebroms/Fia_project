
from scripts.data_preprocessing.loader.factory import load_data
from scripts.data_preprocessing.Target_Features.ClassLabel_Selector import classlabel_selector
from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor

import pandas as pd

from KNN.Classificatore_Knn import Classificatore_KNN

from scripts.Model_Evaluation.Validation.validation_factory import validation_factory


if __name__ == "__main__":
    
    

    dataset = load_data()  # Carica i dati assegnandoli a un pandas dataframe
   
    dataset = Df_Processor.elimina_colonne(dataset) #elimina le colonne che non si desiderano
    
    dataset = Df_Processor.crea_dummy_variables(dataset) #converte le colonne che sono del tipo string in valori numerici usando le dummy variables

    Features, colonne_label = classlabel_selector(dataset) #divide il dataframe in due sotto dataframe: feature e label

    Features = Df_Processor.gestisci_valori_mancanti(Features)
    
    Df_Processor.scala_features(Features)

    print(Features.dtypes)
    print(Features)
    print(colonne_label)


    #-------------------------------parte per testare Holdout


    validators=validation_factory.getvalidationstrategy()
    validators.validation(Features,colonne_label)



    '''print("feature_train \n")
    print(len(Features_train_set))
    print(Features_train_set)
    print("feature_test \n")
    print(len(Features_test_set))
    print(Features_test_set)
    print("label_train \n")
    print(Labels_train_set)
    print(len(Labels_train_set))
    print("label_test \n")
    print(len(Labels_test_set))
    print(Labels_test_set)'''





    #--------------------------------parte aggiunntiva a titolo di prova
    
    #inizializzazione del valore dei k vicini da usare per il classificatore
    #k=input("Inserire il valore dei k vicini da voler usare per costruire il Classificatore KNN: ")

    # Supponiamo che il DataFrame si chiami Features
    '''num_righe = len(Features)
    train_size = int(num_righe * 0.9)  # Calcola il 70% delle righe

    # Creazione dei dataset di training e test
    Features_train_set = Features.iloc[:train_size]  # Primi 70%
    Features_test_set = Features.iloc[train_size:]  # Ultimi 30%

    Labels_train_set = colonne_label.iloc[:train_size]  # Primi 70%
    Labels_test_set = colonne_label.iloc[train_size:]  # Ultimi 30%

    Knn=Classificatore_KNN(Features_train_set,Labels_train_set)

    lista_predizioni=Knn.predizione(Features_test_set)

    print(lista_predizioni)

    print(Labels_test_set)

    c = 0
    for predizione, valore in zip(lista_predizioni, Labels_test_set.iloc[:, 0]):
        if predizione == valore:
            c += 1

    print(c)'''



   
   
    


