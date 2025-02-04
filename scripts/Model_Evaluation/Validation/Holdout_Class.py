import numpy as np

from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation


from ..Metrics.Classe_Metriche import Metriche



from scripts.interfaccia_utente import interfaccia_utente

class HoldoutValidation(validation):
    """
    Implementazione della validazione Holdout.

    Questa classe suddivide il dataset in training set e test set e utilizza un 
    classificatore KNN per effettuare le predizioni e calcolare le metriche di valutazione.
    """


    def __init__(self):

        """
        Inizializza un'istanza di HoldoutValidation e imposta la percentuale del dataset da usare come test set.

        Attributi:
            test_size (float): Percentuale del dataset da assegnare al test set, ottenuta dall'utente.
        """

        

        self.test_size = interfaccia_utente.get_size_of_test()
        print(f"Impostata la percentuale al {self.test_size * 100}%")


    def validation(self, features, target):
        """
        Esegue la validazione Holdout suddividendo il dataset e calcolando le metriche.

        Args:
            features (pd.DataFrame): DataFrame contenente solo le feature del dataset.
            target (pd.DataFrame): DataFrame contenente la classe target.

        Returns:
            list[dict]: Una lista contenente un dizionario con le metriche calcolate.

        Fasi della validazione:
        1. Suddivisione del dataset:
            - Gli indici dei campioni vengono divisi casualmente in training e test set.
            - Il test set ha dimensione pari a `test_size * numero_totale_campioni`.
        2. Predizione con KNN:
            - L'utente seleziona il numero di vicini `k` tramite interfaccia.
            - Il classificatore KNN viene addestrato con il training set.
            - Effettua la predizione sui dati di test.
        3. Calcolo delle metriche:
            - Le predizioni vengono confrontate con le classi reali.
            - Viene generata e plottata la matrice di confusione.
            - Vengono calcolate le metriche selezionate dall'utente.
        """




        num_campioni = len(features) #calcola il numero di campioni (ovvero le righe presenti nel dataframe delle features)
        
        indici_test = np.random.choice(num_campioni, size=int(self.test_size * num_campioni), replace=False)
        
        #La funzione random.choice moltiplica il numero di campioni per la percentuale dichiarata
        #in ingresso in modo da definire quanti record assegnare a test. Inoltre, impostando il 
        #parametro replace a false ogni record può essere selezionato una sola volta. 
        

        indici_training = list(set(range(num_campioni)) - set(indici_test)) 
        #viene fatta una lista che contiene gli indici dei campioni da usare come training ottenuta sottraendo
        #a tutti gli indici quelli selezionati come test set


        X_training, X_test = features.iloc[indici_training], features.iloc[indici_test]
        #individua in features le righe aventi indici del training e aventi gli indici del test
        #e le assegna ai dataframe X_training e X_test che contengono

        Y_training, Y_test = target.iloc[indici_training], target.iloc[indici_test]
        #individua in target le righe aventi indici del training e aventi gli indici del test
        #e le assegna ai dataframe X_training e X_test che contengono
        
        
        k=interfaccia_utente.get_k_neighbours() #chiama la classe interfaccia_utente per far scegliere k
        knn = Classificatore_KNN(X_training, Y_training,k) #crea un istanza di Classificatore
        lista_predizioni = knn.predizione(X_test) 
        #chiama la funzione predizione sull'istanza di classificatore e assegna la predizione delle label del test
        #a una lista
        
        

        lista_label=Y_test.iloc[:, 0].tolist() 
        #mette i valori del dataframe contenente le label dei campioni
        #usati come test in una lista.
        




        Metrica= Metriche(lista_label, lista_predizioni)
        #crea un istanza della classe metriche passando
        #le due liste appena assegnate
        

        Metriche_selezionate = interfaccia_utente.get_metrics_to_calculate()
        #chiama la classe interfaccia_utente per scegliere le metriche da calcolare
        
        lista_matrix=[]#lista vuota per inserire la matrice di confusione
        confusion_matrix=Metrica.make_confusion_matrix()#crea la matrice di confusione con la funzione apposita
        lista_matrix.append(confusion_matrix)#aggiunge alla lista la matrice di confusione appena creata

        Metrica.plot_all_confusion_matrices(lista_matrix)
        #chiama la funzione plot_all_confusion_matrices 


        Metriche_Calcolate=Metrica.calcola_metriche(Metriche_selezionate)
        #chiama la funzione calcola_metriche della classe Metriche per calcolare le metriche appena selezionate
        return [Metriche_Calcolate] #restituisce una lista contenente il dizionario 
        







