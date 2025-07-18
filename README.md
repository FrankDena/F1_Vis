# Progetto per il corso di Visualizzazione delle Informazioni (A.A. 2024-2025)
## F1 Sector Analysis ##

Il sistema offre agli utenti la possibilità di scegliere uno specifico anno, un weekend di gara, la sessione e un pilota e interagire con tre tipi di visualizzazione:
1. __Visualize Single Lap__, per il pilota selezionato l'utente può visualizzare la modulazione della percentuale di pressione dell'acceleratore e le zone di frenata (con lunghezza e delta di velocità dall'ingreso all'uscita) sul giro più veloce per il primo settore della pista;
2. __Compare Fastest Lap__, l'utente può selezionare due piloti e vedere il confronto, nei rispettivi giri più veloci per entrambi, la modulazione della percentuale di pressione dell'acceleratore e le zone di frenata;
3. __Compare Compounds__, per il pilota selezionato l'utente può confrontare la modulazione della percentuale di pressione dell'acceleratore e le zone di frenata per i giri più veloci rispetto ad ogni mescola utilizzata durante la gara.

Il progetto è sviluppato in Python, con l'ausilio di Streamlit per la creazione dell'interfaccia grafica e della libreria FastF1 per i dati tecnici. A partire dall'homepage, la selezione di una specifica visualizzazione esegue uno script diverso.

La costruzione dei plot segue la seguente logica:

+ Si selezionano anno, weekend, sessione e pilota e i dati della telemetria per il primo settore vengono caricati dalla libreria FastF1 e salvati in cache;
+ Dai dati della telemetria si ricavano dei segmenti tra posizioni consecutive (rappresentate dalle coordinate x e y) che verranno utilizzati per disegnare il tracciato e colorarlo, tramite una colormap, in base ai valori percentuali di pressione dell'acceleratore;
+ Lo stesso procedimento viene eseguito per individuare i punti di inizio e di fine frenata, che saranno colorati di bianco per visualizzare le zone di frenata.

Una demo del progetto è accessibile al seguente URL: https://f1plotter.streamlit.app/ o è possibile eseguirlo in locale clonando la repository, installando le dipendenze necessarie elencate nel file *requirements.txt* ed eseguendo il comando:
```
streamlit run Homepage.py
```
da terminale posizionandosi all'interno della directory del progetto. Si aprirà automaticamente una finestra del browser predefinito con l'interfaccia.
