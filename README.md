# Classificazione con k-NN

## Descrizione del Progetto

Questo progetto implementa un classificatore **k-Nearest Neighbors (k-NN)** per la classificazione di dataset generici. Il programma permette di:
- Caricare un dataset contenente dati etichettati.
- Effettuare la pulizia e la trasformazione dei dati con diverse strategie.
- Validare il modello utilizzando **Holdout, Random SubSampling e K-fold Cross Validation**.
- Valutare le prestazioni del modello attraverso metriche dettagliate.
- Salvare i risultati in un file Excel per un'analisi successiva.

L'obiettivo del progetto è fornire un modello di machine learning flessibile e altamente configurabile per diverse tipologie di dataset.

---

## Come eseguire il codice:

1. Bisogna caricare il file contenente il dataset nella cartella dati. Il file deve essere in una delle seguenti estensioni supportate per il caricamento:
- `.csv` o `.tsv`

- `.xlsx` o `.xlx`

- `.json`

- `.txt`

- `.xml`

Il contenuto del file viene caricato in un Pandas DataFrame.

2. L'utente può:

- **Preprocessare il dataset**:
   - Selezionando le colonne da eliminare per migliorare le prestazioni del modello.
   - Selezionando la colonna che vuole come label
   - Scegliendo il metodo per gestire i valori mancanti: **media, mediana o moda**.
   - Applicando **normalizzazione o standardizzazione** alle feature numeriche.
In automatico, invece, se ci saranno colonne contenenti dati di tipo stringa, il codice creerà delle nuove colonne dummy (una per ogni stringa diversa) eliminando quella originale.
   

- **Selezionare il metodo di validazione**:
   - **Holdout**: Viene richiesto all'utente di inserire 1 parametro: la percentuale di campioni del dataset da inserire nel test set. Divide il dataset in training e test set secondo tale percentuale.
   - **Random SubSampling**: Viene richiesto all'utente di inserire 2 parametri: la percentuale di campioni del dataset da inserire nel test set e il numero di esperimenti n. Esegue più divisioni casuali del dataset secondo la percentuale inserita e viene attuato per n volte.
   - **K-fold Cross Validation**: Viene richiesto all'utente di inserire 1 parametro: K, numero di subset in cui dividere il dataset. Suddivide il dataset in K parti per una validazione più robusta, eseguendo K ripetizioni in cui ogni volta viene usato un diverso subset come testset e K-1 subsets come train set.

3. **Configurare il Classificatore k-NN**:
   - Specificando il numero di vicini **(k)** da considerare per la classificazione. Il classificatore predice le label dei campioni di test. 
   - Visualizzare:
       - la **matrice di confusione**: una tabella che mostra il numero di predizioni corrette e errate suddivise per classe, fornendo un'analisi dettagliata delle prestazioni del modello in termini di true positive, true negative, false positive e false negative.
       - la **ROC Curve**: Un grafico che rappresenta la capacità del modello di discriminare tra classe positiva e negativa, mostrando il rapporto tra il tasso di veri positivi (Sensitivity) e il tasso di falsi positivi (1 - Specificity) a diversi livelli di soglia di classificazione.

    Questi elementi vengono costruiti per permettere all'utente di analizzare il comportamento del modello sul test set.

4. **Calcolare e salvare le metriche di valutazione**:
   - Selezionando le metriche che si vogliono calcolare, per valutare le predizioni rispetto alle vere label dei campioni di test.
   - I risultati vengono salvati in un file Excel nella cartella `risultati/`.

---

## Metriche di Valutazione

Il progetto permette la selezione di diverse metriche per valutare le prestazioni del classificatore:

- **Accuracy**: Percentuale di predizioni corrette.
  - Formula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Error Rate**: Percentuale di predizioni errate.
  - Formula: `Error Rate = 1 - Accuracy`
- **Sensitivity (Recall)**: Capacità del modello di identificare correttamente le istanze positive.
  - Formula: `Sensitivity = TP / (TP + FN)`
- **Specificity**: Capacità del modello di identificare correttamente le istanze negative.
  - Formula: `Specificity = TN / (TN + FP)`
- **Geometric Mean (G-Mean)**: Misura dell’equilibrio tra Sensitivity e Specificity.
  - Formula: `G-Mean = sqrt(Sensitivity × Specificity)`
- **Area Under The Curve**: L'area sotto la ROC Curve. Misura la capacità del modello di distinguere tra classe positiva e negativa.

Nel caso di Random SubSampling e Kfold validation, in cui la validazione viene effettuata tante volte quanti sono gli esperimenti e i subsets selezionati, le metriche possono essere calcolate sia per singoli esperimenti sia come valore medio sulle più iterazioni. Nel caso di Holdout verranno calcolate le singole metriche selezionate senza effettuare una scelta. 

---

## Struttura del Progetto

<details>
  <summary>Visualizza la struttura del progetto</summary>

  ```plaintext
  📂 Fia_project
|
├── 📂 dati
|   ├── version_1.csv
|   ├── version_2.xlsx
|   ├── version_3.txt
|   ├── version_4.json
|   ├── version_5.tsv
|
├── 📂 risultati
|   ├── perfromance_model.xlsx
|
├── 📂 scripts
|
|   ├── 📂 data_preprocessing 
|   |   ├── 📂 loader
|   |   |   ├── classe_loader.py
|   |   |   ├── csv_loader.py
|   |   |   ├── Excel_loader.py
|   |   |   ├── factory_loader.py
|   |   |   ├── json_loader.py
|   |   |   ├── txt_loader.py
|   |   |   ├── xml_loader.py
|   |   ├── 📂 pulizia_dataset
|   |   |   ├── pulizia_dataset.py
|   |   ├── 📂 Target_Features
|   |   |   ├── ClassLabel_Selector.py
|   |   
|   ├── 📂 KNN
|       ├── Classificatore_Knn.py
|
|   ├── 📂 Model_Evaluation
|   |   ├── 📂 Metrics
|   |   |   ├── Classe_Metriche.py
|   |   |   ├── visualizzazione_performance.py
|   |   
|   |   ├── 📂 Validation
|   |   |   ├── classe_validation.py
|   |   |   ├── Holdout_Class.py
|   |   |   ├── Kfold_Class.py
|   |   |   ├── Random_Subsampling_Class.py 
|
├── 📂 tests
|   ├── mock_interfaccia_utente.py
|   ├── mock_standardization.py
|   ├── test_Classificatore_KNN.py
|   ├── test_df_proc.py
|   ├── test_Kfold_2.py
|   ├── test_metriche.py
|   ├── test_standardization.py
|
├── .gitignore
|
├── Dockerfile
|
├── main.py
|
├── README.md
|
├── requirements.txt
  ```
</details>

---

## Installazione e Setup

1. **Clonare il repository**:
   ```sh
   git clone <repository-url>
   cd Fia_project
   ```
2. **Installare le dipendenze**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Eseguire lo script principale**:
   ```sh
   python main.py
   ```
   In cui viene eseguita la pipeline appena descritta, chiamando le varie classi responsabili di ogni fase. Verrà utilizzato il dataset di default `version_1.csv`.
   Per usare un dataset personalizzato:
   ```sh
   python main.py --input dati/<nome_file>
   ```



---

## Esecuzione con Docker

È possibile eseguire il progetto tramite Docker:

1. **Costruire l'immagine**:
   ```sh
   docker build -t knn_classifier .
   ```
2. **Eseguire il container**:
   ```sh
   docker run -it -v $(pwd)/dati:/usr/src/app/dati -v $(pwd)/risultati:/usr/src/app/risultati knn_classifier python main.py
   ```

---

## Esempio di Output
Di seguito un esempio di output generato dal modello:

### File Excel generato
Il modello salva automaticamente i risultati delle metriche di valutazione in un file Excel di default chiamato **`risultati/performance_model.xlsx`** ma che può essere anche generato dall'utente stesso. Questo file contiene le seguenti metriche per ogni esperimento (o la media nel caso di più esperimenti se venisse selezionata):

| Metrica        | Esperimento 1 | Esperimento 2 | ... |
|---------------|--------------|--------------|-----|
| Accuracy      | 0.92         | 0.89         | ... |
| Error Rate    | 0.08         | 0.11         | ... |
| Sensitivity   | 0.91         | 0.88         | ... |
| Specificity   | 0.93         | 0.90         | ... |
| Geometric Mean| 0.92         | 0.89         | ... |


### Grafici generati
Il modello genera automaticamente le immagini delle confusion matrix e delle ROC curve, tramite matplotlib, che l'utente può salvare come file.

---

## Contributi

Se vuoi contribuire al progetto:
1. **Forka il repository** e crea una branch per le modifiche.
2. **Invia una pull request** con le modifiche proposte.
3. **Apri un’issue** se trovi bug o hai suggerimenti.

Ogni contributo è benvenuto per migliorare e ottimizzare questo progetto!











