from .Random_Subsampling_CLass import RandomSubsamplingValidation
from .Holdout_Class import HoldoutValidation
from .Kfold_Class import KfoldValidation

from .classe_validation import validation

from scripts.interfaccia_utente import interfaccia_utente
class validation_factory:
    @staticmethod
    def getvalidationstrategy():
        """
        Permette all'utente di scegliere una strategia di validazione, ripetendo la richiesta in caso di errore.
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

        



