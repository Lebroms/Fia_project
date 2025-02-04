from .Random_Subsampling_CLass import RandomSubsamplingValidation
from .Holdout_Class import HoldoutValidation
from .Kfold_Class import KfoldValidation

from .classe_validation import validation

from scripts.interfaccia_utente import interfaccia_utente
class validation_factory:
    """
    Factory per la creazione di strategie di validazione.

    Questa classe fornisce un metodo statico per selezionare e restituire un'istanza 
    della strategia di validazione scelta dall'utente. 

    Methods:
        getvalidationstrategy():
            Richiede all'utente di scegliere una strategia di validazione tra quelle disponibili
            e restituisce l'istanza corrispondente.

    """

    @staticmethod
    def getvalidationstrategy():
        """
        Permette all'utente di scegliere una strategia di validazione.

        Il metodo richiede all'utente di selezionare una delle strategie disponibili e restituisce
        un'istanza della classe corrispondente. Se l'input non è valido, viene richiesto nuovamente.

        Strategie disponibili:
            1 → HoldoutValidation
            2 → RandomSubsamplingValidation
            3 → KfoldValidation

        Returns:
            validation: Un'istanza della classe di validazione scelta.

        Note:
            - Il metodo utilizza `interfaccia_utente.get_validation_method()` per ottenere l'input dell'utente.
            - Se l'utente inserisce un valore non valido, viene mostrato un messaggio di errore e la richiesta viene ripetuta.
        """


        validators = {
            '1': HoldoutValidation,
            '2': RandomSubsamplingValidation,
            '3': KfoldValidation,
        }

        while True:  
            strategy=interfaccia_utente.get_validation_method()

            validator_class = validators.get(strategy)

            if validator_class:
                return validator_class()  # ✅ Restituisce l'istanza scelta
            else:
                print("Opzione non valida. Scegliere un validatore tra quelli elencati.")

        



