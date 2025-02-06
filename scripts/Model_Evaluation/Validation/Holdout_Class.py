import numpy as np

from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation

from ..Metrics.Classe_Metriche import Metriche

class HoldoutValidation(validation):
    """
    Implementazione della validazione Holdout.

    Questa classe suddivide il dataset in training set e test set e utilizza un 
    classificatore KNN per effettuare le predizioni e calcolare le metriche di valutazione.
    """

    def __init__(self,test_size, k):

        """
        Inizializza un'istanza di HoldoutValidation e imposta la percentuale del dataset da usare come test set.

        Attributi:
            test_size (float): Percentuale del dataset da assegnare al test set, ottenuta dall'utente.
            k (int): Numero di vicini da usare nel Classificatore
        """

        

        self.test_size = test_size
        self.k=k


    def validation(self, features, target, metriche_selezionate):
        """
        Esegue la validazione Holdout suddividendo il dataset e calcolando le metriche.

        Args:
            features (pd.DataFrame): DataFrame contenente solo le feature del dataset.
            target (pd.DataFrame): DataFrame contenente la classe target.
            metriche_selezionate (list of str): Lista di stringhe numeriche corrispondenti alle metriche
                                                da calcolare

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
       
        lista_matrix=[] #lista vuota per inserire la matrice di confusione
        liste_di_punti=[]

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
               
        
        knn = Classificatore_KNN(X_training, Y_training,self.k) #crea un istanza di Classificatore
        
        lista_predizioni,_ = knn.predizione_max(X_test)
        _,lista_perc_of_pos=knn.predizione_max(X_test) 
        #chiama la funzione predizione_max sull'istanza di classificatore due volte:
        #1:calcola la predizione delle label dei campioni del test basandosi sulla label maggioritaria 
        # che trova tra i k vicini per i singoli campioni del test e li mette in una lista
        #2:restituisce una lista in cui c'è la percentuale di positivi tra i kvicini per ogni campione del test

        dict_predizioni_con_threshold=knn.predict_label_by_threshold(lista_perc_of_pos)
        #chiama la funzione predizione sull'istanza di classificatore e assegna la predizione delle label del test
        #a una lista

        lista_label=Y_test.iloc[:, 0].tolist() 
        #mette i valori del dataframe contenente le label dei campioni
        #usati come test in una lista.
        
        Metrica= Metriche(lista_label, lista_predizioni)
        #crea un istanza della classe metriche passando
        #le due liste appena assegnate
        
        confusion_matrix=Metrica.make_confusion_matrix()#crea la matrice di confusione con la funzione apposita
        lista_matrix.append(confusion_matrix)#aggiunge alla lista la matrice di confusione appena creata

        Metriche_Calcolate=Metrica.calcola_metriche(metriche_selezionate)
        #chiama la funzione calcola_metriche della classe Metriche per calcolare le metriche appena selezionate
        
        lista_punti=Metrica.costruzione_punti_roc_curve(dict_predizioni_con_threshold)
        liste_di_punti.append(lista_punti)
        
        
        return [Metriche_Calcolate], lista_matrix, liste_di_punti 
        #restituisce una lista contenente il dizionario, una lista contenente una sola matrice 
        #(numpy array 2x2), una lista contenente una sola lista di tuple: ogni tupla è composta da due elementi
        #coordinate x e y di un punto sul grafico della roc curve
        







