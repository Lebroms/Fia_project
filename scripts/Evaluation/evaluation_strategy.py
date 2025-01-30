from abc import ABC, abstractmethod
import numpy as np
import random


#interfaccia comune per tutte le strategie di validazione
class evaluationstrategy(ABC):
    """
        Metodo astratto per gestire la validazione.

        Args:
            X: Il dataframe contenente solo le features
            y: Il dataframe contenente solo la class label

        
        """

    @abstractmethod
    def evaluate(self, X, y, knn_model):
        pass  # Ogni strategia dovrà implementare questo metodo


