import numpy as np
import random
import pandas as pd



class Classificatore_KNN:
    """
    Classe che implementa il classificatore
    """
    def __init__(self, X_train, Y_train, k):
        """
        Costruttore della classe:
            istanzia un oggetto della classe classificatore_KNN che prende in ingresso
            i parametri e li assegna come attributi dell'oggetto istanziato
                
            Args:
                - k (int): numero di vicini per la predizione 
                - X_train (pandas dataframe): training set
                - Y_train (pandas dataframe): labels del training
        """

        self.k=k
        self.X_train_set = X_train
        self.Y_train_set = Y_train

    
    def __distanza_euclidea(self,x1, x2):

        """
        Calcola la distanza euclidea 
            Args:
                - x1 (np array)
                - x2 (np array)
            Return: distanza (np float64)
            """
        

        return np.sum((x1 - x2) ** 2)



    def __trova_k_vicini(self,X_train_set, Y_train_set, X_test, k=3):
        """ 
        Calcola la lista delle label corrispondenti ai primi k vicini per un record di test
            Args:
                - X_train (pandas dataframe): training set
                - Y_train (pandas dataframe): labels del training
                - X_test (np array): singola riga delle features del test set
                - k (int): numero di k più vicini
            Return:
                - k_vicini (list): lista di interi
        
        """
        distanze = []

        for idx, row in X_train_set.iterrows():
            
            dist = self.__distanza_euclidea(row.values, X_test)
            
            distanze.append((dist, int(Y_train_set.loc[idx].values)))
            

        distanze.sort(key=lambda x: x[0])  # ordina la lista delle distanze tra il campione e i vari record in ordine crescente, rispetto alla distanza

        k_vicini = [label for _, label in distanze[:k]]  # riporta la lista delle prime k-label, ordinata rispetto alle distanze
            

        return k_vicini


    def __predici_label_max(self,k_vicini):
        """
        Conta e restituisce quale label (0 o 1) è più presente in k_vicini. In caso di pareggio 
        sceglie la label randomicamente
            Args: 
                - k_vicini (list)
            Returns:
                - random.choice(label_candidate) (int): se non c'è pareggio restituisce direttamente
                                                       il valore della label più presente
        """
        count = {}
        
        for label in k_vicini:
            
            
            if label in count:
                count[label] += 1
            else:
                count[label] = 1

        max_count = max(count.values())
        label_candidate = [label for label, counter in count.items() if counter == max_count]

        return random.choice(label_candidate)
    
    def __calc_perc_pos_in_k_neighbours(self,k_vicini):
        count_of_pos=0
        list_of_perc_pos=[]
        
        for label in k_vicini:
            if label==1:
                count_of_pos+=1
        
        perc_of_pos=(count_of_pos/len(k_vicini))*100

        return perc_of_pos
        

    def predizione_max(self, X_test):
        """
        Calcola la label predetta per ogni campione di test

        Parameters
        ----------
        X_test : pandas Dataframe
            Campioni del test set (solo features).

        Returns
        -------
        list
            lista delle label predette di ogni campione del test set.

        """
        predizione = []
        list_of_perc_pos=[]
        for x_test in np.array(X_test):
            
            
            k_vicini = self.__trova_k_vicini(self.X_train_set, self.Y_train_set, x_test, self.k)
            perc_of_pos=self.__calc_perc_pos_in_k_neighbours(k_vicini)
            list_of_perc_pos.append(perc_of_pos)
            label = self.__predici_label_max(k_vicini)
            predizione.append(label)
        
        return [float(x) for x in predizione],list_of_perc_pos
        
        
        
    def predict_label_by_threshold(self,list_of_perc_pos):
        dict_di_liste={}
        
        for threshold in list(range(0, 111, 10)):
            
            lista_label_con_threshold=[]
            
            
            for perc in list_of_perc_pos:
                
                if perc>=threshold:
                    lista_label_con_threshold.append(1)
                    
                    
                else:
                    lista_label_con_threshold.append(0)
            
                
            dict_di_liste[f"threshold:{threshold}%"]=lista_label_con_threshold
            
            


        return dict_di_liste



