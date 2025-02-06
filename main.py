
from scripts.data_preprocessing.loader.factory import load_data
from scripts.data_preprocessing.Target_Features.ClassLabel_Selector import Class_label_selector
from scripts.data_preprocessing.pulizia_dataset.pulizia_data import Df_Processor

from scripts.interfaccia_utente import interfaccia_utente

from scripts.Model_Evaluation.Validation.Holdout_Class import HoldoutValidation
from scripts.Model_Evaluation.Validation.Random_Subsampling_CLass import RandomSubsamplingValidation
from scripts.Model_Evaluation.Validation.Kfold_Class import KfoldValidation


from scripts.Model_Evaluation.Metrics.Classe_Metriche import Metriche
from scripts.Model_Evaluation.Metrics.visualizzazione_performance import salva_metriche_su_excel

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

    print(Features)
    print(colonne_label)
 
    validation_strategy=interfaccia_utente.get_validation_method()

    if validation_strategy=='1':
        test_size=interfaccia_utente.get_size_of_test()
        print(f"Impostata la percentuale al {test_size * 100}%")
        k=interfaccia_utente.get_k_neighbours()
        print(f"Impostato il numero di vicini k = {k}")
        print("\nCalcolando la predizione...")

        Holdout=HoldoutValidation(test_size,k)

        metriche_selezionate=interfaccia_utente.get_metrics_to_calculate()

        
        lista_metriche,lista_matrici,liste_di_punti=Holdout.validation(Features,colonne_label,metriche_selezionate)
        

    elif validation_strategy=='2':
        num_exp=interfaccia_utente.get_num_experiments()
        print(f"Impostato il numero di esperimenti a {num_exp}")

        test_size=interfaccia_utente.get_size_of_test()
        print(f"Impostata la percentuale al {test_size * 100}%")

        k=interfaccia_utente.get_k_neighbours()
        print(f"Impostato il numero di vicini k = {k}")
        print("\nCalcolando la predizione...")


        metriche_selezionate=interfaccia_utente.get_metrics_to_calculate()

        modalità=interfaccia_utente.get_mod_calculation_metrics(num_exp)


        Random=RandomSubsamplingValidation(num_exp,test_size,k,modalità)
        lista_metriche,lista_matrici,liste_di_punti=Random.validation(Features,colonne_label,metriche_selezionate)

    
    
    elif validation_strategy=='3':
        num_folds=interfaccia_utente.get_num_folds()
        print(f"Impostato il numero di fold a {num_folds}")

        k=interfaccia_utente.get_k_neighbours()
        print(f"Impostato il numero di vicini k = {k}")
        print("\nCalcolando la predizione...")

        metriche_selezionate=interfaccia_utente.get_metrics_to_calculate()

        modalità=interfaccia_utente.get_mod_calculation_metrics(num_folds)

        Kfold=KfoldValidation(num_folds,k,modalità)

        lista_metriche,lista_matrici,liste_di_punti=Kfold.validation(Features,colonne_label,metriche_selezionate)

        

    Metriche.plot_all_confusion_matrices(lista_matrici)

    auc=interfaccia_utente.want_auc_value()
    Metriche.plot_roc_curves(liste_di_punti,auc)


    percorso_completo=interfaccia_utente.get_file()
    salva_metriche_su_excel(lista_metriche,percorso_completo)



    


   
   
    


