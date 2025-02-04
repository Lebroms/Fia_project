import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 


class Metriche:
    def __init__(self, y_real, y_pred):

        """
        Costruttore della classe:
        Istanzia un oggetto della classe metriche contenente come attribuiti y_real e y_pred e come metodi tutte le funzioni di calcolo 
        
        Parametri:
        y_real: Lista dei valori reali delle Label
        y_pred: Lista delle predizioni del modello
        """
        self.y_real = np.array(y_real)  # Converti in array NumPy
        self.y_pred = np.array(y_pred)  # Converti in array NumPy
    def accuracy(self):
        """
        Calcola l'Accuracy Rate.
        Accuracy = (TruePositive + TrueNegative) / (TP + TN + FalsePositive + FalseNegative)
        """

        true_positive,true_negative,false_positive,false_negative,total=self.confusion_matrix()

        accuracy_rate=(true_negative+true_positive)/total
        
        
        return accuracy_rate
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
        true_positive,true_negative,false_positive,false_negative,_=self.confusion_matrix()

        
        #cambiare dataframe da 2 e 4 , a 0 e 1 con funzione
        
        sensitivity=true_positive/(true_positive+false_negative) if (true_positive + false_negative) != 0 else 0

        # vengono iterati simultaneamente i valori reali e le predizioni del modello nel caso in cui i valori reali
        # siano uguali a 1; se sia il valore reale che quello predetto sono uguali a 1, allora verrà aggiunto 1
        # al conteggio dei true positive, altrimenti viene aggiunto 1 al conteggio dei falsi negativi
        return sensitivity

    def specificity(self):
        """
        Calcola il rapporto tra i veri negativi e la somma dei veri negativi e dei falsi positivi.
        Questo misura quanto bene il modello identifica i casi negativi.
        Specificity = TrueNegative / (TrueNegative + FalsePositive)
        """
        true_positive,true_negative,false_positive,false_negative,_=self.confusion_matrix()
        specificity=true_negative/(true_negative+false_positive) if (true_negative + false_positive) != 0 else 0

        # vengono iterati simultaneamente i valori reali e le predizioni del modello nel caso in cui i valori reali
        # siano uguali a 0; se sia il valore reale che quello predetto sono uguali a 0, allora verrà aggiunto 1
        # al conteggio dei true negative, altrimenti viene aggiunto 1 al conteggio dei false positive
        return specificity

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
    
    

    def confusion_matrix (self):
        true_positive=np.sum((self.y_pred == 1) & (self.y_real == 1)) 
        true_negative=np.sum((self.y_pred == 0) & (self.y_real == 0)) 
        
        false_positive=np.sum((self.y_pred == 1) & (self.y_real == 0)) 
        false_negative=np.sum((self.y_pred == 0) & (self.y_real == 1)) 
        total=true_positive+true_negative+false_positive+false_negative

        
        

        return true_positive,true_negative,false_positive,false_negative,total


    def make_confusion_matrix(self):
        # Ottieni i valori dalla funzione confusion_matrix
        true_positive, true_negative, false_positive, false_negative, _ = self.confusion_matrix()

        # Definisci la matrice di confusione correttamente
        confusion_matrix = np.array([[true_negative, false_positive],[false_negative, true_positive]])

        return confusion_matrix

        



    def plot_all_confusion_matrices(self, conf_matrices):
        """
        Plotta più matrici di confusione in un'unica immagine con dimensioni fisse e leggibili.

        Args:
            conf_matrices (list of np.array): Lista di matrici di confusione 2x2.
        """

        # Controllo se la lista è vuota
        if not conf_matrices:
            print("Errore: Nessuna matrice di confusione fornita.")
            return

        num_plot = len(conf_matrices)  # Numero totale di matrici

        # **Limitiamo a 9 esperimenti massimo**
        if num_plot > 9:
            print("Errore: Sono supportati al massimo 9 esperimenti per il plot.")
            return

        cols = min(3, num_plot)  # Al massimo 3 colonne per riga
        rows = (num_plot + cols - 1) // cols  # Calcola il numero di righe necessarie

        # **Figura di dimensione fissa per una buona leggibilità**
        figsize = (15, 10)  # Dimensione della figura in pollici
        fig = plt.figure(figsize=figsize)

        # **Usiamo un layout `gridspec` per garantire che ogni subplot abbia la stessa dimensione**
        gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.4, hspace=0.4)

        labels = ["Negativo", "Positivo"]

        for idx, cm in enumerate(conf_matrices):
            ax = fig.add_subplot(gs[idx])  # Assegna il subplot a una posizione nella griglia
            ax.imshow(cm, cmap="coolwarm")

            # Aggiunge i numeri nelle celle
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, color='black')

            # Imposta gli assi
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predetto")
            ax.set_ylabel("Reale")

            ax.set_title(f"Esperimento {idx + 1}")

        # **Ottimizza la spaziatura tra i subplot**
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()
