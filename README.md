
Il progetto implementa un programma che addestra e valuta un classificatore knn che classifica i tumori come benigni (0) o maligni (1). Il programma prende in ingresso un file di log dove è presente il dataset e premette all'utente di valutare le performance del modello tramite tre metodi di validazione: Holdout, Random SubSampling e K-fold Cross Validation. Le performance del metodo scelto vengono salvate su un file Excel.


Per eseguire il progetto bisogna: \n 
	- fare un clone del repository \n 
	- installare i requirements nel proprio ambiente python \n 
	- dal terminale spostarsi nella directory clonata e digitare python main.py, in questo modo l'applicazione prenderà come file di input del dataset il file version_1.csv \n 
	- se si vuole eseguire con un altro file di input basta salvarlo nella sottodirectory dati/ e digitare sul terminale python main.py --input dati/<nome_file_del_dataset>
	  l'applicazione legge i file di log con estensione csv, xlsx, txt, json e tsv \n 
	- digitare gli input richiesti dal terminale \n 

(Da aggiornare quando si implementa il dockerfile)