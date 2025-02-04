import pandas as pd

import numpy as np

import random 
from ...KNN.Classificatore_Knn import Classificatore_KNN

from .classe_validation import validation
from ...KNN.scegli_k import scegli_k

from scripts.Model_Evaluation.Metrics.Classe_Metriche import Metriche

from scripts.Model_Evaluation.Metrics.scegli_mod_calcolo_metrics import scegli_metriche,scegli_modalita_calcolo_metriche





class RandomSubsamplingValidation(validation):
    """

    Questa classe ha lo scopo di dividere il dataset in due parti: training set e test set.
    Implementa un metodo astratto della classe padre validation.
    
    """

    def __init__(self):

        '''
        Costruttore della classe:
        Istanzia un oggetto della classe RandomSubsamplingValidation che gestisce il modello di validazione 
        Random sub Sampling con attributi "num_experiments" ovvero in numero di volte incui dividere il dataset
        in test set e train set, e "test_size" ovvero la percentuale del dataset da usare come test set. Entrambi i valori 
        inseriti dall'utente tramite riga di comando e assegnati all'istanza

        '''

        while True:  # Ciclo per chiedere il valore di num_experiments finché non è valido
            num_experiments = input("Imposta il numero di esperimenti da eseguire (numero intero positivo)")

            if not num_experiments:  # Se l'utente preme Invio senza inserire nulla
                num_experiments = "10"  # Imposta il valore predefinito
                print("Nessun valore inserito. Imposto numero esperimenti a 10 di default.")


            try:
                num_experiments = int(num_experiments)
                if num_experiments > 0:
                    break  # Esce dal ciclo se il valore è valido
                else:
                    print("Errore: Il valore deve essere positivo. Riprova.")
            except ValueError:
                print("Errore: Inserisci un numero valido (es. 10). Riprova.")

        self.num_experiments = int(num_experiments)
        print(f"Impostato il numero di esperimenti a {self.num_experiments}")

        while True:  # Ciclo per chiedere il valore finché non è valido
            test_size = input(
                "Imposta la percentuale di campioni del dataset da assegnare al test set (valore tra 0 e 1): ").strip()

            if not test_size:  # Se l'utente preme Invio senza inserire nulla
                test_size = "0.2"  # Imposta il valore predefinito
                print("Nessun valore inserito. Imposto test_size a 0.2 di default.")

            test_size = test_size.replace(",", ".")  # Sostituisce la virgola con il punto

            try:
                test_size = float(test_size)  # Converte in float
                if 0 < test_size < 1:
                    break  # Esce dal ciclo se il valore è valido
                else:
                    print("Errore: Il valore deve essere compreso tra 0 e 1. Riprova.")
            except ValueError:
                print("Errore: Inserisci un numero valido (es. 0.2 o 0,2). Riprova.")

        self.test_size = test_size
        print(f"Impostata la percentuale al {test_size * 100}%")




        




    def validation(self, features, target):

        """
        metodo importato dalla classe padre validation 

        Serve per dividere il dataset in training e test in base all'attributo della classe test_size e 
        un numero di volte pari all'attributo num_experiments

        In questo metodo viene chiamata la classe esterna Classificatore_Knn per eseguire la predizione
        sul test_set. E viene chiamata anche la classe Metriche per gestire il calcolo delle metriche 
        

        Parametri:
        features: DataFrame delle features.
        target: DataFrame con le class label.
        """

        num=self.num_experiments #assegna l'attributo num_experiments a num
        
        k=scegli_k() #chiama la funzione scegli_k per scegliere i k vicini
        
        
        

        Metriche_Selezionate =scegli_metriche() #chiama la funzione scegli_metriche 

        modalità=scegli_modalita_calcolo_metriche(num) 
        #chiama la funzione scegli_modalita_calcolo_metriche

        lista_metriche=[]
        



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

            lista_label=y_test.iloc[:, 0].tolist()##mette i valori del dataframe contenente le label dei campioni
            #usati come test in una lista.

            



            
            Metrica= Metriche(lista_label, lista_predizioni)
            #crea un istanza della classe metriche passando
            #le due liste appena assegnate

            Metriche_Calcolate=Metrica.calcola_metriche(Metriche_Selezionate)
            #chiama la funzione calcola_metriche della classe Metriche per calcolare le metriche appena selezionate

            lista_metriche.append((f"Esperimento{i+1}",Metriche_Calcolate))
            
            
            

            
            
            #blocco di codice per verificare il valore di modalità in base alla scelta dell'utente 
            #di calcolare la media delle metriche sugli esperimenti o le metriche per i singoli esperiemnti
            #if modalità == False:#l'utente ha scelto le singole metriche 
                #print(f"Le metriche selezionate per l'esperimento {i+1} valgono:")
                #for c, v in Metriche_Calcolate.items():
                    #print(f"  - {c}: {v:.4f}")

                
            #else:#l'utente ha scelto la media delle metriche 
                #dizionario_metriche[f"Esperimento{i+1}"]=Metriche_Calcolate
                #aggiunge al dizionario_metriche il dizionario delle metriche dell'esperimenti corrente 


        

        if modalità: #uscito dal for questo se modalità è True è un dict di dict
            metriche_raccolte = {}

            # Passo 1: Raccogliamo i valori di ogni metrica
            for _, metriche in lista_metriche:
                for nome_metrica, valore in metriche.items():
                    if nome_metrica not in metriche_raccolte:
                        metriche_raccolte[nome_metrica] = []  # Creiamo una lista per raccogliere i valori
                    metriche_raccolte[nome_metrica].append(valore)

            # Passo 2: Calcoliamo la media per ogni metrica
            medie_metriche = {metrica: np.mean(valori) for metrica, valori in metriche_raccolte.items()}
            return [medie_metriche]

        


        else:
            metriche_per_esperimento=[]
            for _,metriche in lista_metriche:
                metriche_per_esperimento.append(metriche)
            return metriche_per_esperimento #fa una lista in cui ogni elemento è un sottodizionario di dizinario_metriche 

                



        




