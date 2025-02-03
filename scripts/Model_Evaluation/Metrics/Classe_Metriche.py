import numpy as np


from scripts.Model_Evaluation.Metrics.scegli_mod_calcolo_metrics import scegli_modalita_calcolo_metriche,scegli_metriche


class Metriche:
    def __init__(self, y_real, y_pred):

        """
        Costruttore della classe:
        Istanzia un oggetto della classe metriche contenente come attribuiti y_real e y_pred e come metodi tutte le funzioni di calcolo 
        
        Parametri:
        y_real: Lista dei valori reali delle Label
        y_pred: Lista delle predizioni del modello
        """
        self.y_real = y_real #Assegna alla variabile di istanza self.y_real la lista dei valori reali (y_real)
        self.y_pred = y_pred #Assegna alla variabile di istanza self.y_pred la lista dei valori reali (y_pred)

    def accuracy(self):
        """
        Calcola l'Accuracy Rate.
        Accuracy = (TruePositive + TrueNegative) / (TP + TN + FalsePositive + FalseNegative)
        """
        
        correct = sum(1 for real, pred in zip(self.y_real, self.y_pred) if real == pred)
        # Zip viene utilizzato per iterare simultaneamente sulle etichette reali (y_real) e sulle etichette predette (y_pred)
        # l'if  verifica se l'etichetta predetta è uguale a quella reale. Se lo è, aggiunge 1 al conteggio.
        
        return correct / len(self.y_pred)
        #dividiamo il numero di predizioni corrette per tutti i valori predetti

    def error_rate(self):
        """
        Calcola l'Error Rate. Il quale è l'opposto dell'Accuracy Rate, identifica la percentuale di errore.
        Error Rate = 1 - Accuracy = (FalsePositive + FalseNegative) / (TP + TN + FalsePositive + FalseNegative)
        """
        return 1 - self.accuracy()

    def sensitivity(self):
        """
        Calcola il rapporto tra i true positive e la somma dei true positive e dei false negative.
        Questo misura quanto bene il modello identifica i casi positivi.
        Sensitivity = TruePositive / (TruePositive + FalseNegative)
        """
        true_positive = sum(1 for real, pred in zip(self.y_real, self.y_pred) if real == 1 and pred == 1)
        #cambiare dataframe da 2 e 4 , a 0 e 1 con funzione
        false_negative = sum(1 for real, pred in zip(self.y_real, self.y_pred) if real == 1 and pred == 0)
        # vengono iterati simultaneamente i valori reali e le predizioni del modello nel caso in cui i valori reali
        # siano uguali a 1; se sia il valore reale che quello predetto sono uguali a 1, allora verrà aggiunto 1
        # al conteggio dei true positive, altrimenti viene aggiunto 1 al conteggio dei falsi negativi
        return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

    def specificity(self):
        """
        Calcola il rapporto tra i veri negativi e la somma dei veri negativi e dei falsi positivi.
        Questo misura quanto bene il modello identifica i casi negativi.
        Specificity = TrueNegative / (TrueNegative + FalsePositive)
        """
        true_negative = sum(1 for real, pred in zip(self.y_real, self.y_pred) if real == 0 and pred == 0)
        false_positive = sum(1 for real, pred in zip(self.y_real, self.y_pred) if real == 0 and pred == 1)
        # vengono iterati simultaneamente i valori reali e le predizioni del modello nel caso in cui i valori reali
        # siano uguali a 0; se sia il valore reale che quello predetto sono uguali a 0, allora verrà aggiunto 1
        # al conteggio dei true negative, altrimenti viene aggiunto 1 al conteggio dei false positive
        return true_negative / (true_negative + false_positive) if (true_negative + false_positive) != 0 else 0

    def geometric_mean(self):
        """
        Calcola la Media Geometrica tra Sensitivity e Specificity.
        Geometric Mean = sqrt(Sensitivity * Specificity)
        """
        sensitivity = self.sensitivity()
        specificity = self.specificity()
        return np.sqrt(sensitivity * specificity)

    '''def auc(self):
        """
        Calcola l'Area Under the Curve (AUC) come la media di Sensitivity e Specificity.
        """
        # Ordina le predizioni in ordine decrescente
        sorted_indices = np.argsort(self.y_pred)[::-1]
        y_real_sorted = self.y_real[sorted_indices]
        y_pred_sorted = self.y_pred[sorted_indices]

        # Calcola TPR e FPR per ogni soglia
        tpr = []
        fpr = []
        num_positive = np.sum(y_real_sorted == 1)
        num_negative = np.sum(y_real_sorted == 0)

        # Inizializza i conteggi
        tp = 0
        fp = 0

        for i in range(len(y_pred_sorted)):
            if y_real_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / num_positive if num_positive != 0 else 0)
            fpr.append(fp / num_negative if num_negative != 0 else 0)

        # Calcola l'area sotto la curva ROC usando la regola del trapezio
        auc_value = 0
        for i in range(1, len(fpr)):
            auc_value += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2

        return auc_value'''
    

    def all_the_above(self):
        '''
        Metodo che richiama tuttu i metodi per calcolare tutte le metriche  
        '''

        accuracy=Metriche.accuracy()
        errore_rate=Metriche.error_rate()
        sensitivity=Metriche.sensitivity()
        specificity=Metriche.specificity()
        geometric_mean=Metriche.geometric_mean()
        area_under_the_curve=Metriche.auc()


    


    def calcola_metriche(self, metriche_scelte):
        """
        Calcola le metriche selezionate dall'utente e restituisce un dizionario con i valori.
        Parametri:
        metriche_scelte= una lista numerica corrispondente alle metriche selezionata 

        Return:
        Restituisce un dizionario di metriche che come chiavi ha i nomi delle metriche calcolate e come valori 
        i corrispondenti risultati

        """
        lista_metriche = [
            "Accuracy", "Error Rate", "Sensitivity", "Specificity",
            "Geometric Mean", "Area Under the Curve", "Tutte le metriche"]

        # Dizionario con i riferimenti alle funzioni (non eseguite subito)
        metrics_functions = {
            "1": self.accuracy,
            "2": self.error_rate,
            "3": self.sensitivity,
            "4": self.specificity,
            "5": self.geometric_mean,
            "7": self.all_the_above
        }

        metriche_calcolate = {}
        if "7" in metriche_scelte:
            for key, func in metrics_functions.items():
                if key != "7":  # Evitiamo di chiamare all_the_above()
                    metriche_calcolate[lista_metriche[int(key)-1]] = func()
        else:
            for key in metriche_scelte:
                nome_chiave = lista_metriche[int(key)-1]  # Usa key come indice per ottenere la stringa
                metriche_calcolate[nome_chiave] = metrics_functions[key]()  # Assegna il valore alla chiave corretta
        
        
        return metriche_calcolate
    
    



    
    
        









