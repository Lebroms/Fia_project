def scegli_k():
    while True:
        k = input(
            "Inserire il valore dei k vicini da voler usare per costruire il Classificatore KNN (default: 3): ").strip()

        if k == "":  # Se l'utente non inserisce nulla
            k = 3  # Imposta il valore di default
            print("Nessun valore inserito. Impostato k = 3 di default.")
            break

        try:
            k = int(k)
            if k <= 0:
                print("Errore: k deve essere un intero positivo. Riprova.")

            else:
                break
        except ValueError:
            print("Errore: Inserisci un numero intero valido. Riprova.")

    print(f"Impostato il numero di vicini k = {k}")

    print("\nCalcolando la predizione...")

    return k