

import numpy as np

import random 

from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation

from scripts.Model_Evaluation.Metrics.Classe_Metriche import Metriche


class RandomSubsamplingValidation(validation):
    """
    Implementazione della validazione Random Subsampling.
    
    Questa classe suddivide il dataset in training set e test set ripetutamente
    per un numero specificato di esperimenti e utilizza un classificatore KNN
    per effettuare le predizioni e calcolare le metriche di valutazione.
    """


    def __init__(self,num_experiments,test_size,k,modalità):


        """
        Inizializza un'istanza di RandomSubsamplingValidation.

        Attributi:
            
            num_experiments (int): Numero di esperimenti di validazione da eseguire.
            
            test_size (float): Percentuale del dataset da assegnare al test set.
            
            k (int): numero di vicini da usare nel Classificatore.
            
            modalità (boolean): True se l'utente vuole visualizzare la media delle metriche selezionate,
                                False se l'utente vuole visualizzare le metriche selezionate per ogni esperimentomodalità (boolean): True se l'utente vuole visualizzare la media delle metriche selezionate,
                                                    False se l'utente vuole visualizzare le metriche selezionate per ogni esperiment           
        """
        self.num_experiments =num_experiments
        self.test_size = test_size
        self.k=k
        self.modalità=modalità



    def validation(self, features, target, metriche_selezionate):

        """
        Esegue la validazione Random Subsampling suddividendo il dataset e calcolando le metriche.

        Args:
            features (pd.DataFrame): DataFrame contenente solo le feature del dataset.
            target (pd.DataFrame): DataFrame contenente la classe target.
            metriche_selezionate (list of str): Lista di stringhe numeriche corrispondenti alle metriche
                                                da calcolare

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
        :param metriche_selezionate:
        """

        lista_metriche=[]#crea un dizionario vuoto in cui salvare le metriche dei vari esperimenti
        lista_matrix=[]#crea una lista vuota in cui mettere le matrici di confusione dei vari esperimenti
        liste_di_punti=[]#crea una lista vuota in cui mettere le liste di punti per costruire una roc curve per ogni esperimento


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

            knn = Classificatore_KNN(X_train, y_train,self.k) #istanzia uno oggetto Classificatore 
            lista_predizioni,_ = knn.predizione_max(X_test)
            _,lista_perc_of_pos=knn.predizione_max(X_test) 
            #chiama la funzione predizione_max sull'istanza di classificatore due volte:
            #1:calcola la predizione delle label dei campioni del test basandosi sulla label maggioritaria 
            # che trova tra i k vicini per i singoli campioni del test e li mette in una lista
            #2:restituisce una lista in cui c'è la percentuale di positivi tra i kvicini per ogni campione del test

            dict_predizioni_con_threshold=knn.predict_label_by_threshold(lista_perc_of_pos)
            #chiama la funzione predizione sull'istanza di classificatore e assegna la predizione delle label del test
            #a una lista

            lista_label=y_test.iloc[:, 0].tolist()#mette i valori del dataframe contenente le label dei campioni
            #usati come test in una lista.

            Metrica= Metriche(lista_label, lista_predizioni)
            #crea un istanza della classe metriche passando
            #le due liste appena assegnate
            
            confusion_matrix=Metrica.make_confusion_matrix()#crea la matrice di confusione con la funzione apposita
            lista_matrix.append(confusion_matrix)#aggiunge alla lista la matrice di confusione appena creata per l'esperimento corrente

            Metriche_Calcolate=Metrica.calcola_metriche(metriche_selezionate)
            #chiama la funzione calcola_metriche della classe Metriche per calcolare le metriche appena selezionate


            lista_punti=Metrica.costruzione_punti_roc_curve(dict_predizioni_con_threshold)
            liste_di_punti.append(lista_punti)
            
            
            lista_metriche.append((f"Esperimento{i+1}",Metriche_Calcolate))
            #appende a lista_metriche una tupla che ha come primo elemento il nome 
            #dell'esperimento e come secondo elemento un dizionario che ha come chiavi le metriche
            #selezionate e come valori le i valori delle metriche sul fold corrente
            #alla fine del for sarà una lista di tuple
            

        if self.modalità: 
            metriche_raccolte = {}

            for _, metriche in lista_metriche:
                for nome_metrica, valore in metriche.items():
                    if nome_metrica not in metriche_raccolte:
                        metriche_raccolte[nome_metrica] = []  #Creiamo una lista per raccogliere i valori della singola metrica tra i vari esperimenti
                    metriche_raccolte[nome_metrica].append(valore)

            #Calcoliamo la media per ogni metrica
            medie_metriche = {metrica: np.mean(valori) for metrica, valori in metriche_raccolte.items()}
            return [medie_metriche],lista_matrix,liste_di_punti 
            #la lista con un dizionario contenente le medie delle metriche sugli esperimenti
            #la lista di numpy array (2x2) rappresentanti le confusion matrix per ognuno degli esperimenti usato come test
            #una lista di liste di tuple: ogni tupla è composte da due elementi
            #coordinate x e y di un punto sul grafico della roc curve

        else:
            metriche_per_esperimento=[]
            for _,metriche in lista_metriche:
                metriche_per_esperimento.append(metriche)
            return metriche_per_esperimento,lista_matrix,liste_di_punti 
        #la lista di dizionari (uno per ciascun esperimento) contenenti i risultati del calcolo delle metriche selezionate
        #la lista di numpy array (2x2) rappresentanti le confusion matrix per ognuno degli esperimenti
        #una lista di liste di tuple: ogni tupla è composte da due elementi
        #coordinate x e y di un punto sul grafico della roc curve



        




