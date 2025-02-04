import pandas as pd

import numpy as np

import random 
from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation


from scripts.Model_Evaluation.Metrics.Classe_Metriche import Metriche



from scripts.interfaccia_utente import interfaccia_utente



class RandomSubsamplingValidation(validation):
    """
    Implementazione della validazione Random Subsampling.
    
    Questa classe suddivide il dataset in training set e test set ripetutamente
    per un numero specificato di esperimenti e utilizza un classificatore KNN
    per effettuare le predizioni e calcolare le metriche di valutazione.
    """

    def __init__(self):

        """
        Inizializza un'istanza di RandomSubsamplingValidation.

        Attributi:
            num_experiments (int): Numero di esperimenti di validazione da eseguire.
            test_size (float): Percentuale del dataset da assegnare al test set.
        """
        self.num_experiments = interfaccia_utente.get_num_experiments()
        print(f"Impostato il numero di esperimenti a {self.num_experiments}")

        
        self.test_size = interfaccia_utente.get_size_of_test()
        print(f"Impostata la percentuale al {self.test_size * 100}%")




        




    def validation(self, features, target):

        """
        Esegue la validazione Random Subsampling suddividendo il dataset e calcolando le metriche.

        Args:
            features (pd.DataFrame): DataFrame contenente solo le feature del dataset.
            target (pd.DataFrame): DataFrame contenente la classe target.

        Returns:
            list[dict]: Una lista contenente i dizionari con le metriche calcolate per ogni esperimento,
                        oppure un unico dizionario con la media delle metriche se selezionato dall'utente.

        Fasi della validazione:
        1. Suddivisione del dataset:
            - Gli indici dei campioni vengono mescolati casualmente.
            - Il test set ha dimensione pari a `test_size * numero_totale_campioni`.
            - Il processo viene ripetuto `num_experiments` volte.
        2. Predizione con KNN:
            - L'utente seleziona il numero di vicini `k` tramite interfaccia.
            - Il classificatore KNN viene addestrato con il training set.
            - Effettua la predizione sui dati di test.
        3. Calcolo delle metriche:
            - Le predizioni vengono confrontate con le classi reali.
            - Viene generata e plottata la matrice di confusione.
            - Vengono calcolate le metriche selezionate dall'utente.
        Se l'utente sceglie di aggregare le metriche, il metodo restituisce la media delle metriche sui vari esperimenti.
        """

        num=self.num_experiments #assegna l'attributo num_experiments a num
        
        k=interfaccia_utente.get_k_neighbours() #chiama la classe interfaccia_utente per far scegliere k
        
        
        

        Metriche_Selezionate =interfaccia_utente.get_metrics_to_calculate() 
        #chiama la classe interfaccia_utente per scegliere le metriche da calcolare 

        modalità=interfaccia_utente.get_mod_calculation_metrics(num) 
        #chiama la classe interfaccia_utente per scegliere le modalità con cui calcolare le metriche

        lista_metriche=[]#crea un dizionario vuoto in cui salvare le metriche dei vari esperimenti
        lista_matrix=[]#crea una lista vuota in cui mettere le matrici di confusione dei vari esperimenti
        



        for i in range(self.num_experiments):  # Ripetiamo per ogni esperimento la divisione del dataset, il calcolo della predizione e delle metriche 
            n = len(features)  # Numero totale di campioni
            n_campioni_test = int(n * self.test_size)  # Numero di campioni per il test set

            #Crea una lista di indici e li mescoliamo
            indici = list(range(n))  # Lista degli indici [0, 1, 2, ..., n-1]
            random.shuffle(indici)  # Mischiamo gli indici casualmente in modo da pescarli randomicamente 

            # Selezioniamo indici per train e test
            indici_test= indici[:n_campioni_test]  # scegliamo i primi indici come indici dei campioni da usare come test  `test_size` indici come test set
            indici_train = indici[n_campioni_test:]  # Il resto sono gli indici dei campioni del training set

            # Creiamo i dataset di train e di test usando gli indici selezionati
            X_train = features.iloc[indici_train]
            y_train = target.iloc[indici_train]
            X_test = features.iloc[indici_test]
            y_test = target.iloc[indici_test]

            knn = Classificatore_KNN(X_train, y_train,k) #istanzia uno oggetto Classificatore 
            lista_predizioni = knn.predizione(X_test) 
            #chiama la funzione predizione per predire la label dei campioni di test 
            # per l'esperimento corrente e le assegna a una lista

            lista_label=y_test.iloc[:, 0].tolist()#mette i valori del dataframe contenente le label dei campioni
            #usati come test in una lista.

            



            
            Metrica= Metriche(lista_label, lista_predizioni)
            #crea un istanza della classe metriche passando
            #le due liste appena assegnate
            
            confusion_matrix=Metrica.make_confusion_matrix()#crea la matrice di confusione con la funzione apposita
            lista_matrix.append(confusion_matrix)#aggiunge alla lista la matrice di confusione appena creata per l'esperimento corrente

            Metriche_Calcolate=Metrica.calcola_metriche(Metriche_Selezionate)
            #chiama la funzione calcola_metriche della classe Metriche per calcolare le metriche appena selezionate

            lista_metriche.append((f"Esperimento{i+1}",Metriche_Calcolate))
            #appende a lista_metriche una tupla che ha come primo elemento il nome 
            #dell'esperimento e come secondo elemento un dizionario che ha come chiavi le metriche
            #selezionate e come valori le i valori delle metriche sul fold corrente
            #alla fine del for sarà una lista di tuple
            
            

        
        Metrica.plot_all_confusion_matrices(lista_matrix)
        #chiama la funzione plot_all_confusion_matrices

        if modalità: 
            metriche_raccolte = {}

            for _, metriche in lista_metriche:
                for nome_metrica, valore in metriche.items():
                    if nome_metrica not in metriche_raccolte:
                        metriche_raccolte[nome_metrica] = []  #Creiamo una lista per raccogliere i valori della singola metrica tra i vari esperimenti
                    metriche_raccolte[nome_metrica].append(valore)

            #Calcoliamo la media per ogni metrica
            medie_metriche = {metrica: np.mean(valori) for metrica, valori in metriche_raccolte.items()}
            return [medie_metriche] #la lista di dizionari contenente le medie delle metriche sugli esperimenti

        


        else:
            metriche_per_esperimento=[]
            for _,metriche in lista_metriche:
                metriche_per_esperimento.append(metriche)
            return metriche_per_esperimento #lista di dizionari delle metriche sui singoli esperimenti

                



        




