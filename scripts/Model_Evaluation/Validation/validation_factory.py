from Random_Subsampling_CLass import RandomSubsamplingValidation
# from Holdout_CLass import HoldoutValidation
# from Kfold_Class import KFoldValidation

from classe_validation import validation


class validation_factory:
    @staticmethod
    def getvalidationstrategy():

        print("\n Scegliere quale delle seguenti strategia di validazione usare: \n \u25BA Premi \U00000031\U0000FE0F\U000020E3 per Holdout \n")
        print("\u25BA Premi \U00000031\U0000FE0F\U000020E3 per Holdout \n")
        print("\u25BA Premi \U00000032\U0000FE0F\U000020E3 per Random Sub Sampling \n") 
        print("\u25BA Premi \U00000033\U0000FE0F\U000020E3 per K-fold Cross Validation \n")


        strategy=input()
        validators = {
            '1': HoldoutValidation,     # Loader per file CSV
            '2':RandomSubsamplingValidation,      #Loader per file Tsv
            '3': KfoldValidation,        # Loader per file Txt
        }

        validator_class=validators.get(strategy)

        if validator_class:
            # Restituisci un'istanza del loader trovato
            return validator_class()
        else:
            # Solleva un'eccezione se l'estensione non è supportata
            raise ValueError(f"Validatore scelto non supportato, riavviare il codice scegliendo un validatore tra quelli elencati")
        



