

# Descrizione del progetto
Il progetto implementa un programma che addestra e valuta un classificatore knn che classifica i tumori come benigni (0) o maligni (1). Il programma prende in ingresso un file di log dove è presente il dataset e permette all'utente di validare il modello tramite tre metodi: Holdout, Random SubSampling e K-fold Cross Validation. Le performance del Classificatore vengono salvate su un file Excel nella cartella risultati.


# Funzionalità principali
L'utente può:
1) caricare qualsiasi tipo di file purchè l'estensione sia supportata. 
2) effettuare la pulizia del dataset scegliendo diverse opzioni:(media,mediana,moda) per la        sostituzione di valori NaN, (normalization,standardization) per lo scaling delle feature e inoltre le colonne che si desiderano eliminare.
3) scegliere il metodo di validazione da adottare tra quelli supportati e scegliere i parametri relativi a ognuno di essi, tra cui il numero di k vicini da usare nel classificatore.
4) visualizzare la matrice di confusione e la ROC curve costruita basandosi su un livello di soglia che aumenta da 0 a 100%.
5) scegliere quali metriche calcolare per valutare le performance del classificatore. Nel caso ci siano più esperimenti effettuati, si possono calcolare la media per ogni metrica o le singole metriche per ogni esperimento. 



# Struttura del Progetto  
📂 Fia_project
|
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

 


# Installazione e Setup

Per eseguire il progetto bisogna:

1) Fare un clone del repository

2) Installare le librerie contenute in requirements.txt nel proprio ambiente python

3) Dal terminale spostarsi nella directory clonata e digitare "python main.py", in questo modo l'applicazione prenderà come file di input del dataset il file version_1.csv

4) Se si vuole eseguire con un altro file di input basta salvarlo nella sottodirectory dati/ e digitare sul terminale "python main.py --input dati/<nome_file_del_dataset>".
L'applicazione legge i file di log con estensione csv, xlsx, txt, json e tsv

5) Digitare gli input richiesti dal terminale



# Guida all’uso con Docker

Inoltre l'applicazione può essere eseguita anche attraverso il Dockerfile da una macchina Linux (con Docker installato):

1) Eseguire un clone del repository

2) Nella bash spostarsi all'interno della cartella clonata

3) Costruire un'immagine dell'applicazione digitando il comando "docker build -t <nome_dell'immagine> ." 

4) Creare ed eseguire il conteiner basato sull'immagine del progetto tramite due Bind Mounts (uno per l'input e uno per l'output) digitando: docker run -it -v /percorso_host/Fia_projects/dati:/usr/src/app/dati -v /percorso_host/Fia_projects/risultati:/usr/src/app/risultati <nome_dell'immagine> python main.py



