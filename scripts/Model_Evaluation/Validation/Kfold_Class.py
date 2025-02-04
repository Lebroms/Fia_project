import numpy as np
from pyexpat import features

from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation



from ..Metrics.Classe_Metriche import Metriche



from scripts.interfaccia_utente import interfaccia_utente


class KfoldValidation(validation):
    """
    Implementazione della validazione K-Fold.

    Questa classe suddivide il dataset in `n_folds` sottogruppi e ripete la validazione
    utilizzando ogni fold come test set una volta, mentre gli altri servono da training set.
    Dopo aver effettuato le predizioni con KNN, calcola e restituisce le metriche.
    """
    def __init__(self):
        """
        Inizializza un'istanza di KfoldValidation e imposta il numero di fold.

        Attributi:
            n_folds (int): Numero di sottogruppi in cui dividere il dataset, scelto dall'utente.
        """

        self.n_folds = interfaccia_utente.get_num_folds()
        print(f"Impostato il numero di fold a {self.n_folds}")



    def validation(self, features, target):
        """
        Esegue la validazione K-Fold suddividendo il dataset e calcolando le metriche.

        Args:
            features (pd.DataFrame): DataFrame contenente solo le feature del dataset.
            target (pd.DataFrame): DataFrame contenente la classe target.

        Returns:
            list[dict]: Una lista di dizionari contenenti le metriche calcolate per ogni fold.

        Fasi della validazione:
        1. Suddivisione del dataset:
            - Gli indici dei campioni vengono mescolati casualmente.
            - Il dataset viene diviso in `n_folds` sottogruppi di dimensioni uguali (o quasi).
        2. Validazione iterativa:
            - Per ogni iterazione, un fold diverso viene usato come test set.
            - Gli altri fold vengono usati come training set.
        3. Predizione con KNN:
            - L'utente seleziona il numero di vicini `k` tramite interfaccia.
            - Il classificatore KNN viene addestrato e testato su ogni fold.
        4. Calcolo delle metriche:
            - Viene generata e plottata la matrice di confusione.
            - Le metriche vengono calcolate per ogni fold e restituite.

        Se l'utente sceglie di aggregare le metriche, il metodo restituisce la media delle metriche sui vari fold.
        """
        num_campioni = len(features) #calcola il numero di campioni (ovvero le righe presenti nel dataframe delle features)

        indici = np.arange(num_campioni)
        #Viene generato un array di indici che va da 0 a num_campioni - 1. Questi indici verranno
        #poi mescolati tramite Shuffle e usati per comporre i vari folds
        


        np.random.shuffle(indici)
        #L'ordine degli indici viene scambiato in modo casuale, così da garantire un
        #risultato più valido per il test. Questo perchè potrebbero esserci casi in cui i valori 
        #di class label sono disposti prima con tutti i 2 e poi con tutti i 4.
        

        fold_sizes = np.full(self.n_folds, num_campioni // self.n_folds, dtype=int)
        fold_sizes[:num_campioni % self.n_folds] += 1
        #Viene generato un array di lunghezza pari al numero di folds, e ogni elemento dell'array
        #indica quanti campioni ci sono in ogni fold. Se il risultato del rapporto tra il numero
        #di campioni e il numero di folds non è un intero il resto dei campioni viene ridistribuito tra 
        #i primi folds.


        k = interfaccia_utente.get_k_neighbours() #chiama la classe interfaccia_utente per far scegliere k
        
         

        num=self.n_folds #assegna l'attributo n_folds a num
        

        Metriche_Selezionate =interfaccia_utente.get_metrics_to_calculate()
        #chiama la classe interfaccia_utente per scegliere le metriche da calcolare 


        modalità = interfaccia_utente.get_mod_calculation_metrics(num)
        #chiama la classe interfaccia_utente per scegliere le modalità con cui calcolare le metriche



        lista_metriche=[]#crea un dizionario vuoto in cui salvare le metriche dei vari esperimenti
        lista_matrix=[]#crea una lista vuota in cui mettere le matrici di confusione dei vari esperimenti

        current = 0
        for index, i in enumerate(fold_sizes):
            indici_test = indici[current:current + i]
            #Seleziona gli indici da 0 a i, che verranno usati come test set."""
            indici_training = np.concatenate((indici[:current], indici[current + i:]))
            #Gli indici non selezionati compongono il training set."""
            feature_test_set = features.iloc[indici_test]
            feature_train_set = features.iloc[indici_training]

            label_test_set = target.iloc[indici_test]
            label_train_set = target.iloc[indici_training]

            
            current += i
            #Incrementa l'indice per la prossima iterazione. In questo modo ogni iterazione 
            #seleziona un pezzo del dataset come test e il resto come training, e quindi alla
            #fine ogni campione verrà utilizzato solo una volta come test.
            



            knn=Classificatore_KNN(feature_train_set,label_train_set,k)
            lista_predizioni=knn.predizione(feature_test_set)
            #chiama la funzione predizione sull'istanza di classificatore e assegna
            #la predizione delle label del test a una lista

            lista_label=label_test_set.iloc[:, 0].tolist()
            #mette i valori del dataframe contenente le label dei campioni
            #usati come test in una lista.


            Metrica= Metriche(lista_label, lista_predizioni)
            #crea un istanza della classe metriche passando
            #le due liste appena assegnate

            
            confusion_matrix=Metrica.make_confusion_matrix()#crea la matrice di confusione con la funzione apposita
            lista_matrix.append(confusion_matrix)#aggiunge alla lista la matrice di confusione appena creata per il fold corrente
            
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
                        metriche_raccolte[nome_metrica] = []  #Creiamo una lista per raccogliere i valori della singola metrica tra i vari fold
                    metriche_raccolte[nome_metrica].append(valore)

            #Calcoliamo la media per ogni metrica
            medie_metriche = {metrica: np.mean(valori) for metrica, valori in metriche_raccolte.items()}
            return [medie_metriche] #la lista di dizionari contenente le medie delle metriche sui fold

        


        else:
            metriche_per_esperimento=[]
            for _,metriche in lista_metriche:
                metriche_per_esperimento.append(metriche)
            return metriche_per_esperimento #lista di dizionari delle metriche sui singoli fold

            



