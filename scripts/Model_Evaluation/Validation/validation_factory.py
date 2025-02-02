from .Random_Subsampling_CLass import RandomSubsamplingValidation
from .Holdout_Class import HoldoutValidation
from .Kfold_Class import KfoldValidation

from .classe_validation import validation


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

        while True:  # 🔄 Continua a chiedere finché l'utente non sceglie un'opzione valida
            print("\n Scegliere quale delle seguenti strategia di validazione usare: \n")
            print("\u25BA Premi \U00000031\U0000FE0F\U000020E3 per Holdout \n")
            print("\u25BA Premi \U00000032\U0000FE0F\U000020E3 per Random Sub Sampling \n")
            print("\u25BA Premi \U00000033\U0000FE0F\U000020E3 per K-fold Cross Validation \n")

            strategy = input("Inserisci il numero della strategia scelta: ").strip()

            validator_class = validators.get(strategy)

            if validator_class:
                return validator_class()  # ✅ Restituisce l'istanza scelta
            else:
                print("Opzione non valida. Scegliere un validatore tra quelli elencati.")

        



