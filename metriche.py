import numpy as np

class Metriche:
    def __init__(self, y_real, y_pred):
        """
        Inizializza la classe con i valori reali (y_real) e le predizioni del modello (y_pred).
        :param y_real: Lista dei valori reali
        :param y_pred: Lista delle predizioni del modello
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

    def auc(self):
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

        return auc_value


    '''def scegli_metrica(self):
        """
        Calcola tutte le metriche e le raccoglie in un dizionario.
        """
        print("\nScegliere tra le seguenti le metriche da scegliere: \n")
        lista_metriche=["Accuracy", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean","Area Under the Curve"]
        for index, el in enumerate(lista_metriche, start=1):  # start=1 per partire da 1 invece di 0
            print(f"\u25BA {index}. Per selezionare {el} premere {index}\n")

        metriche_scelte=(input)


        metrics = {
            "Accuracy": self.accuracy(),
            "Error Rate": self.error_rate(),
            "Sensitivity": self.sensitivity(),
            "Specificity": self.specificity(),
            "Geometric Mean": self.geometric_mean(),
            "Area Under the Curve": self.auc()
        }
        return metrics'''







