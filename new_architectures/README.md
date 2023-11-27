Questo file contiene le istruzioni per creare i dataset per le reti Pix2Pix e cycleGAN e testare le architetture contenute in questo repository

1) Loggarsi su yoga con profilo intertwin e selezionare il kernel "pytorch" per l'esecuzione di tutti i notebook. Potrebbe essere necessario installare manualmente, eseguendo dei comandi "pip install", le seguenti librerie: astropy, requests, PyJWT, Nds2utils, cryptography.

PER LA CREAZIONE DEI DATASET

2) Aprire il notebook "centered_images".
OSS: il file .tar.gz contenente i dati .h5 si può reperire al link 
"
https://unipiit-my.sharepoint.com/personal/a012554_unipi_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fa012554%5Funipi%5Fit%2FDocuments%2FSharedDocsOneDriveRazzano%2FInterTwin%2Ddati%2Daux&ga=1
"
l'unico file che ci serve è quello di nome "O3a_SL_aux.tar.gz", io essendo pigro ho scaricato il file e caricato direttamente nel mio ambiente yoga, posizionandolo in una cartella di nome data nella stessa directory madre in cui si trova il notebook, dunque in esso non c'è un codice per scaricare il file ma solo per estrarre l'archivio.
3) Nella prima cella del codice, assicurarsi che alla prima esecuzione del codice la variabile "already_extracted" sia impostata su False, poi cambiarla a True.
4) Le prossime sezioni hanno lo scopo di illustrare il processo di creazione delle immagini e sono anche abbastanza ben commentate a mio avviso, eseguire tutte le celle fino alla sezione "Save the images locally".
5) Nella cella sotto la suddetta sezione assicurarsi di nuovo che alla prima esecuzione la variabile "already_done" sia impostata su False, poi cambiarla a True. Questa è la procedura che richiederà più tempo di tutte in quanto eseguirà tutte le q-scan centrate.
6) La sezione successiva crea un immagine a puro scopo dimostrativo.
7) Le ultime due sezioni creano i dataset veri e propri. Possiamo scegliere in entrambi i casi il numero del canale ausiliario che per cui vogliamo creare il dataset, io per ora l'ho sempre creato col numero 2 in quanto ad occhio era il canale ausiliario più correlato al canale di strain, se si vuole creare un dataset con un canale ausiliario diverso consiglio di cambiare anche il nome della cartella del dataset (contenuto nella variabile "ch_img_dir")
OSS: Anche per il dataset della cycleGAN si creano immagini accoppiate strain/aux in quanto ci sarà comodo durante la valutazione della GAN dove sarà necessario fare un confronto tra l'immagine generata e l'immagine reale.

PER IL TRAINING E IL TESTING DELLE ARCHITETTURE

OSS: Tutte le informazioni seguenti valgono per entrambe le reti.
8) Il codice delle reti funziona a patto che la cartella contenente i dati abbia il nome corretto e sia nel posto giusto. Il nome è lo stesso che ho dato nel notebook precedente per la creazione del dataset, quindi 'aux_channel_two' per la rete Pix2Pix e 'aux_channel_two_cycle' per la cycleGAN. Tali cartelle dovranno essere poste nella stessa cartella madre in cui si trovano quelle contenenti tutti i file delle reti. Se dovessimo dare un nome diverso alle cartelle contenenti i dati dovremmo cambiare questo nome solo alla variabile "dataset_folder_name" all'interno del file dataset.py.
9) L'esecuzione del training avviene tramite il notebook "main.ipynb". Si noti che il notebook è predisposto per caricare di default l'ultimo checkpoint disponibile nella cartella save_states, io ho caricato per entrambe le reti il checkpoint più avanzato di cui disponevo, se si volesse far ripartire il training da zero occorrerebbe per esempio rimuovere questo file dalla suddetta cartella.
10) Tramite il notebook "test GAN.ipynb" è possibile invece vedere in tempo reale (anche durante il training) i risultati della generazione di immagini partendo da dati nel dataset sia di training che di testing. Anche questo notebook caricherà automaticamente l'ultimo checkpoint che trova nella cartella save_states.
