import streamlit as st
import pandas as pd
import fastf1 as ff1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import KDTree
import re

ff1.Cache.enable_cache('cache') # Abilita la cache per velocizzare il caricamento dei dati

def load_weekends_for_year(year):
    # Restituisce la lista degli eventi per un anno specifico
    # da mostrare nella sidebar
    schedule = ff1.get_event_schedule(year)
    return schedule['EventName'].tolist()


def load_drivers_for_session(year, wknd, ses):
    # Restituisce la lista dei piloti per una sessione specifica
    session = ff1.get_session(year, wknd, ses)
    session.load()
    driver_numbers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in driver_numbers]
    return drivers

# --- Sidebar per controlli ---
st.sidebar.title("F1 Telemetry Viewer")
year = st.sidebar.selectbox("Anno", [2022, 2023, 2024], index=2)
wknd = st.sidebar.selectbox("Weekend", load_weekends_for_year(year), index=0)
ses = st.sidebar.selectbox("Sessione", ['FP1', 'FP2', 'FP3', 'Q', 'R'], index=4)
driver = st.sidebar.selectbox("Pilota", load_drivers_for_session(year, wknd, ses))

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

    dists = corners['Distance'].values #Estrai le distanze delle curve dalla partenza, .values è necessario per ottenere un array numpy

    if not sector.empty:
        # Trova gli indici delle curve selezionate nel settore
        idx_first = corners[corners['Number'] == sector.loc[0, 'Number']].index[0] #idx_first è l'indice della prima curva del settore all'interno di corners
        idx_last = corners[corners['Number'] == sector.loc[len(sector)-1, 'Number']].index[0] # idx_last è l'indice dell'ultima curva del settore all'interno di corners
    else:
        print("Nessun settore trovato con le curve specificate.")
        exit()

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

    brake = sector_telemetry['Brake']
    # Trova gruppi contigui dove Brake == 1
    groups = (brake != brake.shift()).cumsum() #ogni blocco contiguo di valori uguali in Brake ottiene un ID di gruppo unico
    brake_groups = sector_telemetry[brake == 1].groupby(groups) #Ogni gruppo corrisponde a una sequenza contigua di Brake == 1

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

    # Per il lap time, estraiamo il tempo del giro e lo formattiamo usando regex
    lap_time = str(lap['LapTime'])[:-3]
    match = re.search(r'(\d+:(\d+:\d+\.\d+))', lap_time)
    if match:
        lap_time = match.group(2)
    fig.suptitle(f"{driver} - {weekend.EventName} - {session.event.year} - {ses} - Turn {sector.loc[0,'Number']}\n Compound: {lap['Compound']} - LAP: {lap['LapNumber'].astype(int)}\n LAP Time: {lap_time}", fontsize=12, fontweight='bold')

    # suptitle imposta il titolo della figura con il nome del pilota, l'evento, l'anno e la sessione e il numero del primo settore

    #Disegno del tracciato nero sullo sfondo
    ax.plot(x,y,color='black', linewidth=10, alpha=0.8, zorder=1) # Disegna il tracciato in nero

    # Recupera la linea appena creata
    linea_tracciato = ax.lines[-1]

    # Imposta cap e join arrotondati
    linea_tracciato.set_solid_capstyle('round')   # estremità arrotondate
    linea_tracciato.set_solid_joinstyle('round')  # giunzioni arrotondate

    # Disegna i segmenti colorati in base all'acceleratore
    norm = mpl.colors.Normalize(color.min(), color.max()) # Normalizza i valori dell'acceleratore tra il minimo e il massimo
    colormap = mpl.colormaps['RdYlGn'] # Colormap per colorare i segmenti in base all'acceleratore
    lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=6) # Crea una collezione di linee con il colormap 'summer_r'
    lc.set_capstyle('round') # Imposta lo stile delle estremità delle linee come arrotondato
    lc.set_array(color) # Imposta i valori dell'acceleratore come array per la collezione di linee
    line = ax.add_collection(lc) # Aggiunge la collezione di linee all'asse

    # cbaxes = fig.add_axes([0.15, 0.05, 0.7, 0.02]) # Aggiunge un asse per la barra dei colori
    # normlegend = mpl.colors.Normalize(vmin=color.min(), vmax=color.max()) # Normalizza i valori dell'acceleratore per la legenda dei colori 
    # legend = mpl.colorbar.ColorbarBase(cbaxes, cmap=colormap,
    #                                 norm=normlegend, orientation='horizontal') # Crea la barra dei colori orizzontale
    # legend.set_label('Acceleratore (%)', fontsize = 12) # Imposta l'etichetta della barra dei colori
    # legend.ax.xaxis.set_label_position('bottom')           # Posizionala sotto la barra
    # legend.ax.xaxis.labelpad = 8                           # Aggiungi spazio tra la barra e l'etichetta

    legend = fig.colorbar(
        lc, ax=ax, orientation='horizontal', fraction=0.03, pad=0.02
    )
    legend.set_label('Acceleratore (%)', fontsize=10)

    # Crea una lista di LineCollection, una per ogni segmento di frenata
    brake_collections = []
    for _, group in brake_groups:
        # Prendi coordinate X, Y per questo gruppo
        xg = group['X'].values
        yg = group['Y'].values

        # Crea punti e segmenti
        points = np.array([xg, yg]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Crea LineCollection per questo segmento
        brake_lc = LineCollection(segments, colors='cyan', linewidths=3)
        brake_lc.set_capstyle('round')  # Imposta estremità arrotondate
        brake_lc.set_linestyle('-')  # Imposta lo stile della linea
        brake_collections.append(brake_lc)

    # Aggiungi ogni segmento di frenata
    for brake_lc in brake_collections:
        ax.add_collection(brake_lc)

    # Punti di frenata con alone bianco (triangolo verso il basso)
    # ax.scatter(brake_pts['X'], brake_pts['Y'],
    #         marker='o', c='white', s=160, label='Frenata', edgecolors='black', linewidths=1.5, zorder=3, alpha=0.65)
    
    centered_brake_points = []
    speed_deltas_for_each_brake_zone = []

    for _, group in brake_groups:
        # Trova indice centrale del gruppo
        center_idx = len(group) // 3

        # Estrai X, Y del punto centrale
        x_center = group.iloc[center_idx]['X']
        y_center = group.iloc[center_idx]['Y']

        # Salva la coppia (X, Y)
        centered_brake_points.append((x_center, y_center))

    
        # Prendi le velocità del gruppo
        speeds = group['Speed'].values

        # Calcola la differenza di velocità tra il primo e l'ultimo punto del gruppo
        if len(speeds) > 1:
            speed_delta = int(speeds[0]) - int(speeds[-1])
            speed_deltas_for_each_brake_zone.append((speeds[0].astype(int), speeds[-1].astype(int), speed_delta))
        else:
            speed_deltas_for_each_brake_zone.append((speeds[0].astype(int),speeds[0].astype(int),0))
 
    def extract_text_from_speed_delta(speed_delta_tuple):
        # Estrae il testo dalla tupla di velocità
        return f"{speed_delta_tuple[0]} Km/h - {speed_delta_tuple[1]} Km/h ({speed_delta_tuple[2]} km/h)"

    # Aggiungi numeri dentro i marker
    for i, (brake_coor_x, brake_coor_y) in enumerate(centered_brake_points, start=1):
        ax.text(brake_coor_x, brake_coor_y, str(i), color='black', fontsize=8,
                ha='center', va='center', zorder=4, alpha=0.7)
        ax.scatter(brake_coor_x, brake_coor_y, marker='o', c='white', s=160, 
                   label='Frenata '+str(i)+": " + extract_text_from_speed_delta(speed_deltas_for_each_brake_zone[i-1]), edgecolors='black', linewidths=1.5, zorder=3, alpha=0.65)

    # ax.scatter(brake_pts['X'], brake_pts['Y'],
    #            marker='v', c='white', alpha=0.4, s=120, zorder=2)

    # Punti di accelerazione (triangolo verso l’alto)
    # ax.scatter(accel_pts['X'], accel_pts['Y'],
    #         marker='o', c='white', s=120, label='Accelerazione', edgecolors='black', linewidths=1.5, zorder=3)

    # ax.scatter(accel_pts['X'], accel_pts['Y'],
    #            marker='^', c='white', alpha=0.4, s=120, zorder=2)





    ax.axis('equal') # Imposta gli assi con la stessa scala per X e Y
    ax.axis('off') # Nasconde gli assi per un aspetto più pulito
    ax.legend(loc='best', frameon=False, fontsize=7, markerscale=0.5,) # Aggiunge la legenda in alto a destra


    # Disegna la freccia

    dx = x.iloc[6] - x.iloc[3]
    dy = y.iloc[6] - y.iloc[3]

    # Normalizza
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length

    # Offset laterale (es. 5 unità)
    offset = 160
    x_offset = nx * offset
    y_offset = ny * offset

    start = (x.iloc[3] + x_offset, y.iloc[3] + y_offset)
    end = (x.iloc[6] + x_offset, y.iloc[6] + y_offset)


    arrow = FancyArrowPatch(
        posA=start, posB=end,
        arrowstyle="-|>", mutation_scale=30, color='black', linewidth=2, zorder=4
    )
    ax.add_patch(arrow)

    # Trovo le curve per disegnarne dopo i numeri
    # Per disegnarli all'esterno della pista devo trovare le curve a quale punto della telemetria sono più vicine
    # Una volta trovato, posso calcolare la direzione della curva e spostare il testo di un offset laterale
    curves = corners.loc[idx_first:idx_last]  # campi x, y, Number, Angle, Distance

    # Crea un albero KD per ricerca veloce nei punti telemetria
    telemetry_points = np.column_stack((tel['X'], tel['Y']))
    tree = KDTree(telemetry_points)

    # Trova gli indici dei punti più vicini alle curve
    curve_coords = np.column_stack((curves['X'], curves['Y']))
    _, indices = tree.query(curve_coords) # indices ora contiene per ogni curva l’indice del punto telemetria più vicino

    # Calcola direzione per ogni curva
    dx = tel['X'].iloc[indices + 1].values - tel['X'].iloc[indices - 1].values
    dy = tel['Y'].iloc[indices + 1].values - tel['Y'].iloc[indices - 1].values

    lengths = np.hypot(dx, dy)
    nx = -dy / lengths
    ny = dx / lengths

    offset = 260

    curves['X_offset'] = curves['X'] + nx * offset
    curves['Y_offset'] = curves['Y'] + ny * offset


    for _, row in curves.iterrows():
        ax.text(
            row['X_offset'], row['Y_offset'],
            f"{row['Number']}", fontsize=10, fontweight='bold',
            color='black', ha='center', va='center',
            bbox=dict(facecolor='none', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
        )



    #plt.show()
    st.pyplot(fig)

