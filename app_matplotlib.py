import streamlit as st
import pandas as pd
import fastf1 as ff1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

st.set_page_config(layout="wide")  # Usa tutta la larghezza del browser

# --- Sidebar per controlli ---
st.sidebar.title("F1 Telemetry Viewer")
year = st.sidebar.selectbox("Anno", [2022, 2023, 2024], index=2)
wknd = st.sidebar.number_input("Weekend (round)", min_value=1, max_value=25, value=5)
ses = st.sidebar.selectbox("Sessione", ['FP1', 'FP2', 'FP3', 'Q', 'R'], index=4)
driver = st.sidebar.selectbox("Pilota", ['HAM', 'VER', 'LEC', 'NOR', 'RUS', 'SAI', 'ALO'])

# --- Carica dati quando si clicca ---
if st.sidebar.button("Carica dati e mostra plot"):
    session = ff1.get_session(year, wknd, ses)
    session.load()
    weekend = session.event

    lap = session.laps.pick_drivers(driver).pick_fastest()
    tel = lap.get_telemetry()
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners

    # Filtra un settore specifico (curve 1-4)
    turns = [1, 2, 3, 4]
    sector = corners[corners['Number'].isin(turns)] # Estrai le curve specificate, costruisci un settore

    if not sector.empty:
        dists = corners['Distance'].values #Estrai le distanze delle curve dalla partenza, .values è necessario per ottenere un array numpy

        # Trova gli indici delle curve selezionate nel settore
        idx_first = corners[corners['Number'] == sector.loc[0, 'Number']].index[0] #idx_first è l'indice della prima curva del settore all'interno di corners
        idx_last = corners[corners['Number'] == sector.loc[len(sector)-1, 'Number']].index[0] # idx_last è l'indice dell'ultima curva del settore all'interno di corners

        # Calcola le distanze di inizio e fine del settore

        d_prev = dists[idx_first-1] if idx_first > 0 else 0 #d_prev è la distanza dalla curva precedente alla prima curva del settore, se esiste
        d0 = dists[idx_first] #d0 è la distanza della prima curva del settore dalla linea di partenza
        d_end0 = dists[idx_last] #d_end0 è la distanza dell'ultima curva del settore dalla linea di partenza
        d_next = dists[idx_last+1] if idx_last < len(dists)-1 else tel['Distance'].max() #d_next è la distanza della curva successiva all'ultima curva del settore, se esiste, altrimenti è la distanza massima della telemetria

        d_start = (d_prev + d0) / 2 #d_start è la distanza di inizio del settore, media tra la curva precedente e la prima curva del settore
        d_end = (d_end0 + d_next) / 2 #d_end è la distanza di fine del settore, media tra l'ultima curva del settore e la curva successiva

        # Telemetria del tratto
        sector_telemetry = tel.loc[(tel['Distance'] >= d_start) & (tel['Distance'] <= d_end)].copy() # Estrai la telemetria del settore tra d_start e d_end

        # Dati originali
        x = sector_telemetry['X'] #x contiene le coordinate X della posizione nel settore specificato del tracciato
        y = sector_telemetry['Y'] #y contiene le coordinate Y della posizione nel settore specificato del tracciato
        color = sector_telemetry['Throttle'] #color contiene i valori dell'acceleratore nel settore specificato

        # Crea punti e segmenti per il tracciato, i segmenti sono le linee che saranno colorate in base all'acceleratore
        points = np.array([x, y]).T.reshape(-1, 1, 2) # crea punti che serviranno per tracciare i segmenti da colorare
        #x e y sono due array numpy di lunghezza uguale: np.array([x, y]) crea una matrice con due righe, una per x e una per y, e un numero di colonne
        #pari alla lunghezza di x e y (N). 
        #.T traspone la matrice, trasformandola in una matrice di N righe e 2 colonne, in cui ogni riga rappresenta un punto (x, y).
        #reshape(-1, 1, 2) cambia la forma della matrice in modo che abbia una dimensione di 2 (x, y) per ogni punto, creando un array di punti 2D.
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        #concatenate unisce i punti in segmenti, creando un array di segmenti 2D, dove ogni segmento è rappresentato da due punti consecutivi.

        # Trova i punti di frenata
        brake_pts = sector_telemetry.loc[sector_telemetry['Brake'].astype(int).diff() == 1, ['X', 'Y']] 
        # Estrai i punti in cui il freno è stato attivato, differenza tra valori consecutivi di 'Brake' per trovare i cambiamenti
        # brake_pts coniente le coordinate X e Y dei punti di frenata

        # Trova i punti di accelerazione
        throttle = sector_telemetry["Throttle"]
        mask_accel_start = (throttle > 5) & (throttle.shift(1) <= 5) # mask_accel_start è una maschera booleana che identifica i punti in cui l'acceleratore supera il 5% dopo essere stato sotto il 5%
        # in mask_accel_start c'è True per i punti in cui l'acceleratore supera il 5% dopo essere stato più basso

        accel_pts = sector_telemetry.loc[mask_accel_start, ['X', 'Y']] # accel_pts contiene le coordinate X e Y dei punti di accelerazione

        fig, ax = plt.subplots(figsize=(6, 6)) # Crea una figura e un asse per il tracciato
        fig.suptitle(f"{driver} - {weekend.EventName} - {session.event.year} - {ses} - Turn {sector.loc[0,'Number']}")
        # suptitle imposta il titolo della figura con il nome del pilota, l'evento, l'anno e la sessione e il numero del primo settore

        #Disegno del tracciato nero sullo sfondo
        ax.plot(x,y,color='black', linewidth=12, label='Tracciato') # Disegna il tracciato in nero

        # Disegna i segmenti colorati in base all'acceleratore
        norm = mpl.colors.Normalize(color.min(), color.max()) # Normalizza i valori dell'acceleratore tra il minimo e il massimo
        colormap = mpl.colormaps['summer_r'] # Colormap per colorare i segmenti in base all'acceleratore
        lc = mpl.LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=6) # Crea una collezione di linee con il colormap 'summer_r'
        lc.set_array(color) # Imposta i valori dell'acceleratore come array per la collezione di linee
        line = ax.add_collection(lc) # Aggiunge la collezione di linee all'asse

        cbaxes = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # Aggiunge un asse per la barra dei colori
        normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max()) # Normalizza i valori dell'acceleratore per la legenda dei colori 
        legend = mpl.colorbar.ColorbarBase(cbaxes, cmap=colormap,
                                        norm=normlegend, orientation='horizontal') # Crea la barra dei colori orizzontale


        # Disegna i punti di frenata e accelerazione
        ax.scatter(brake_pts['X'],  brake_pts['Y'],
                c='red',   s=80, label='Inizio frenata',    zorder=2)
        ax.scatter(accel_pts['X'], accel_pts['Y'],
                c='green', s=80, label='Inizio accelerazione', zorder=2)

        ax.axis('equal') # Imposta gli assi con la stessa scala per X e Y
        ax.axis('off') # Nasconde gli assi per un aspetto più pulito
        ax.legend(loc='upper right', frameon=True) # Aggiunge la legenda in alto a destra con il frame visibile

        plt.show()