from abc import ABC, abstractmethod
import numpy as np
import random


#interfaccia comune per tutte le strategie di evaluation
class validation(ABC):
    """
        Metodo astratto per gestire la validazione.

        Args:
            X: Il dataframe contenente solo le features
            y: Il dataframe contenente solo la class label

        
        """

    @abstractmethod
    def validation(self, features, target):
        pass  # Ogni strategia dovrà implementare questo metodo


