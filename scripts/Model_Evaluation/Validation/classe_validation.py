from abc import ABC, abstractmethod
import numpy as np
import random


#interfaccia comune per tutte le strategie di evaluation
class validation(ABC):
    """
    Classe astratta per la gestione delle strategie di validazione dei modelli di machine learning.

    Questa classe definisce un'interfaccia comune per tutte le strategie di validazione, 
    forzando l'implementazione del metodo `validation` nelle classi derivate.

    Methods:
        validation(features, target):
            Metodo astratto che deve essere implementato nelle sottoclassi. 
            Definisce la logica di validazione su un dataset.
    
    """

    @abstractmethod
    def validation(self, features, target):
        """
        Metodo astratto per eseguire la validazione del modello.

        Questo metodo deve essere implementato nelle classi derivate per eseguire una specifica strategia 
        di validazione

        Args:
            features (pd.DataFrame o np.array): Il dataset contenente le feature.
            target (pd.Series o np.array): Il dataset contenente la classe target.

        Returns:
            Variabile definita nella sottoclasse: Può essere l'accuratezza del modello, un insieme di metriche, 
            un dizionario con i risultati della validazione o qualsiasi altro output utile alla valutazione del modello.
        """

        pass  # Ogni strategia dovrà implementare questo metodo


