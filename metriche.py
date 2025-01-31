import numpy as np

class Metriche:
    def __init__(self, y_true, y_pred):
        """
        Inizializza la classe con i valori reali (y_true) e le predizioni del modello (y_pred).
        :param y_true: Lista dei valori reali
        :param y_pred: Lista delle predizioni del modello
        """
        self.y_true = y_true #Assegna alla variabile di istanza self.y_true la lista dei valori reali (y_true)
        self.y_pred = y_pred #Assegna alla variabile di istanza self.y_pred la lista dei valori reali (y_pred)

    def accuracy(self):
        """
        Calcola l'Accuracy Rate.
        Accuracy = (TruePositive + TrueNegative) / (TP + TN + FalsePositive + FalseNegative)
        """
        correct = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == pred)
        # per iterare simultaneamente sulle etichette reali (y_true) e sulle etichette predette (y_pred)
        # l'if  verifica se l'etichetta predetta è uguale a quella reale. Se lo è, aggiunge 1 al conteggio.
        return correct / len(self.y_true)
        #dividiamo il numero di predizioni corrette per la lunghezza totale del dataset

    def error_rate(self):
        """
        Calcola l'Error Rate. Il quale è l'opposto dell'Accuracy Rate, identifica la percentuale di errore.
        Error Rate = 1 - Accuracy
        """
        return 1 - self.accuracy()

    def sensitivity(self):
        """
        Calcola il rapporto tra i veri positivi e la somma dei veri positivi e dei falsi negativi.
        Questo misura quanto bene il modello identifica i casi positivi.
        Sensitivity = TruePositive / (TruePositive + FalseNegative)
        """
        true_positive = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == 1 and pred == 1)
        false_negative = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == 1 and pred == 0)
        # vengono iterati simultaneamente i valori reali e le predizioni del modello, se coincidono, allora viene
        # aggiunto 1 al conteggio dei veri positivi, altrimenti viene aggiunto 1 al conteggio dei falsi negativi
        return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

    def specificity(self):
        """
        Calcola il rapporto tra i veri negativi e la somma dei veri negativi e dei falsi positivi.
        Questo misura quanto bene il modello identifica i casi negativi.
        Specificity = TrueNegative / (TrueNegative + FalsePositive)
        """
        true_negative = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == 0 and pred == 0)
        false_positive = sum(1 for true, pred in zip(self.y_true, self.y_pred) if true == 0 and pred == 1)
        # vengono iterati simultaneamente i valori reali e le predizioni del modello, se coincidono, allora viene
        # aggiunto 1 al conteggio dei veri negativi, altrimenti viene aggiunto 1 al conteggio dei falsi positivi
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
        sensitivity = self.sensitivity()
        specificity = self.specificity()
        return (sensitivity + specificity) / 2  # Media di Sensitivity e Specificity


    def calculate_all_metrics(self):
        """
        Calcola tutte le metriche e le raccoglie in un dizionario.
        """
        metrics = {
            "Accuracy": self.accuracy(),
            "Error Rate": self.error_rate(),
            "Sensitivity": self.sensitivity(),
            "Specificity": self.specificity(),
            "Geometric Mean": self.geometric_mean(),
            "Area Under the Curve": self.auc()
        }
        return metrics

    def collect_metrics_for_all_points(self):
        """
        Raccoglie i valori delle metriche per ogni coppia di (y_true, y_pred)
        """
        # Lista per raccogliere i record
        metrics_list = []

        # Itera su tutte le coppie di y_true e y_pred
        for true_value, pred_value in zip(self.y_true, self.y_pred):
            # Calcola le metriche per ogni singola coppia (true_value, pred_value)
            self.y_true = [true_value]
            self.y_pred = [pred_value]
            metrics = self.calculate_all_metrics()

            # Aggiungi il record alla lista
            metrics["y_true"] = true_value
            metrics["y_pred"] = pred_value
            metrics_list.append(metrics)






