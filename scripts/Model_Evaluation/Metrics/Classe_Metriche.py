import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 





class Metriche:
    """
    Classe per il calcolo delle metriche di valutazione di un modello di classificazione binaria.

    Questa classe fornisce metodi per calcolare l'accuracy, il tasso di errore, la sensibilità,
    la specificità, la media geometrica e altre metriche di valutazione, oltre a generare la matrice
    di confusione e plot delle matrici di confusione per più esperimenti.

    """
    def __init__(self, y_real, y_pred):

        """
        Inizializza un oggetto della classe Metriche con i valori reali e predetti.

        Attributi:
            y_real (list o np.array): Lista o array dei valori reali della variabile target.
            y_pred (list o np.array): Lista o array delle predizioni del modello.
        """
        self.y_real = np.array(y_real)  # Converti in array NumPy
        self.y_pred = np.array(y_pred)  # Converti in array NumPy
    def accuracy(self):
        """
        Calcola l'Accuracy Rate.

        Formula:
            Accuracy = (True Positive + True Negative) / (Totale)

        Returns:
            float: Il valore dell'accuracy.
        """

        true_positive,true_negative,false_positive,false_negative,total=self.confusion_matrix()

        accuracy_rate=(true_negative+true_positive)/total
        
        
        return accuracy_rate
        #dividiamo il numero di predizioni corrette per tutti i valori predetti

    def error_rate(self):
        """
        Calcola l'Error Rate, ovvero il complemento dell'Accuracy Rate.

        Formula:
            Error Rate = 1 - Accuracy

        Returns:
            float: Il valore dell'error rate.
        """
        
        return 1 - self.accuracy()

    def sensitivity(self):
        """
        Calcola la Sensitivity.

        Misura la capacità del modello di identificare correttamente i positivi.

        Formula:
            Sensitivity = True Positive / (True Positive + False Negative)

        Returns:
            float: Il valore della sensitivity (da 0 a 1). Restituisce 0 se il denominatore è = 0
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
        Calcola la Specificity.

        Misura la capacità del modello di identificare correttamente i negativi.

        Formula:
            Specificity = True Negative / (True Negative + False Positive)

        Returns:
            float: Il valore della specificity (da 0 a 1). Restituisce 0 se il denominatore è = 0
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

        Formula:
            Geometric Mean = sqrt(Sensitivity * Specificity)

        Returns:
            float: Il valore della geometric mean.
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
        """
        Calcola tutte le metriche disponibili e le restituisce in un dizionario.

        Returns:
            dict: Dizionario con tutte le metriche calcolate.
        """
        return {
            "Accuracy": self.accuracy(),
            "Error Rate": self.error_rate(),
            "Sensitivity": self.sensitivity(),
            "Specificity": self.specificity(),
            "Geometric Mean": self.geometric_mean()
            
        }


    


    def calcola_metriche(self, metriche_scelte):
        """
        Calcola le metriche selezionate dall'utente e restituisce un dizionario con i valori.

        Args:
            metriche_scelte (list of str): Lista di stringhe numeriche corrispondenti alle metriche da calcolare.

        Returns:
            dict: Dizionario con i nomi delle metriche selezionate come chiavi e i rispettivi valori.
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
            return self.all_the_above()
        else:
            for key in metriche_scelte:
                nome_chiave = lista_metriche[int(key)-1]  # Usa key come indice per ottenere la stringa
                metriche_calcolate[nome_chiave] = metrics_functions[key]()  # Assegna il valore alla chiave corretta
        
        
        return metriche_calcolate
    
    

    def confusion_matrix (self):
        """
        Calcola la matrice di confusione per il modello di classificazione binaria.

        Returns:
            tuple: Una tupla contenente i seguenti valori:
                - true_positive (int): Numero di veri positivi (TP).
                - true_negative (int): Numero di veri negativi (TN).
                - false_positive (int): Numero di falsi positivi (FP).
                - false_negative (int): Numero di falsi negativi (FN).
                - total (int): Numero totale di campioni.
        """
        true_positive=np.sum((self.y_pred == 1) & (self.y_real == 1)) 
        true_negative=np.sum((self.y_pred == 0) & (self.y_real == 0)) 
        
        false_positive=np.sum((self.y_pred == 1) & (self.y_real == 0)) 
        false_negative=np.sum((self.y_pred == 0) & (self.y_real == 1)) 
        total=true_positive+true_negative+false_positive+false_negative

        
        

        return true_positive,true_negative,false_positive,false_negative,total


    def make_confusion_matrix(self):
        """
        Costruisce e restituisce la matrice di confusione nel formato standard 2x2.

        La matrice è organizzata come segue:
            [[True Negative (TN), False Positive (FP)]
            [False Negative (FN), True Positive (TP)]]

        Returns:
            np.array: Matrice di confusione 2x2.
        """
        # Ottieni i valori dalla funzione confusion_matrix
        true_positive, true_negative, false_positive, false_negative, _ = self.confusion_matrix()

        # Definisci la matrice di confusione correttamente
        confusion_matrix = np.array([[true_negative, false_positive],[false_negative, true_positive]])

        return confusion_matrix

        



    def plot_all_confusion_matrices(self, conf_matrices):
        """
        Plotta più matrici di confusione in un'unica immagine, organizzandole in una griglia.

        Args:
            conf_matrices (list of np.array): Lista di matrici di confusione 2x2 da visualizzare.

        Comportamento:
            - Se la lista è vuota, viene stampato un messaggio di errore.
            - Supporta un massimo di 9 esperimenti per garantire una buona leggibilità.
            - Organizza i plot in un massimo di 3 colonne.
            - Utilizza un layout `gridspec` per garantire che ogni subplot abbia la stessa dimensione.

        Il colore delle celle è determinato dalla colormap "coolwarm", con i valori numerici sovrapposti.
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
