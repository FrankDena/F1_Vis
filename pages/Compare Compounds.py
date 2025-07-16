import re
import streamlit as st
import pandas as pd
import fastf1 as ff1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import KDTree
import math

# ! -- Nello script non si considera più il giro veloce in assoluto, ma si considerano i giri più veloci per ogni compound -- !

# Abilita la cache per velocizzare il caricamento dei dati
ff1.Cache.enable_cache('cache')

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

# Questa funzione disegna i grafici per ogni giro considerato prima
# e li visualizza in Streamlit
# La funzione cicla su ogni giro del dizionario fastest_per_compound
def plot_on_axis(ax, obj):
    lap = obj['fastest_lap'] # Estrai il giro più veloce dal dizionario
    tel = lap.get_telemetry() # Estrai la telemetria del giro
    # Filtra la telemetria del settore tra d_start e d_end
    mask = (tel['Distance']>=d_start)&(tel['Distance']<=d_end) # Maschera boolean per filtrare la telemetria del settore
    
    sec_tel = tel[mask] # Telemetria del settore filtrata
    brk = sec_tel['Brake']
    groups = (brk != brk.shift()).cumsum() # Raggruppa i punti di frenata, creando un gruppo per ogni cambiamento del valore del freno
    brake_groups = sec_tel[brk == 1].groupby(groups) # Raggruppa i punti di frenata per i gruppi creati
    brake_pts = sec_tel.loc[sec_tel['Brake'].astype(int).diff() == 1, ['X','Y']] # Estrai il primo punto di ogni gruppo di frenata
    brake_collections = []
    for _, group in brake_groups: # Cicla su ogni gruppo di frenata
        xg = group['X'].values
        yg = group['Y'].values

        points = np.array([xg, yg]).T.reshape(-1, 1, 2) # Crea un array di punti per il gruppo di frenata
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        brake_lc = LineCollection(segments, color='white', linewidths = 3)
        brake_lc.set_capstyle('round')  # Imposta le estremità arrotondate
        brake_lc.set_linestyle('-')
        brake_collections.append(brake_lc) # Aggiungi la collezione di linee di frenata alla lista

    centered_brake_points = []
    speed_deltas_for_each_brake_zone = []

    for _, group in brake_groups: # Cicla su ogni gruppo di frenata
        center_idx = len(group) // 3
        x_center = group['X'].iloc[center_idx]
        y_center = group['Y'].iloc[center_idx]
        centered_brake_points.append((x_center, y_center)) # Aggiungi il punto di frenata centrato alla lista
        speeds = group['Speed'].values
        if len(speeds) > 1:
            speed_delta = int(speeds[0]) - int(speeds[-1]) # Calcola la differenza di velocità tra il primo e l'ultimo punto del gruppo di frenata
            speed_deltas_for_each_brake_zone.append((speeds[0].astype(int), speeds[-1].astype(int), speed_delta))
        else:
            speed_deltas_for_each_brake_zone.append((speeds[0].astype(int),speeds[0].astype(int),0))

    def extract_text_from_speed_delta(speed_delta_tuple):
        # Estrae il testo dalla tupla di velocità
        return f"{speed_delta_tuple[0]} Km/h - {speed_delta_tuple[1]} Km/h ({speed_delta_tuple[2]} km/h)"
     
    x, y = tel.loc[mask,'X'], tel.loc[mask,'Y'] # x e y contengono le coordinate X e Y della posizione nel settore specificato del tracciato ed ottenuto dalla maschera
    thr = tel.loc[mask,'Throttle'] # thr contiene i valori dell'acceleratore nel settore specificato
    # segmenti colorati
    pts = np.array([x,y]).T.reshape(-1,1,2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.plot(x,y, color='black', linewidth=10, alpha=0.8, zorder=1)
    # Recupera la linea appena creata
    linea_tracciato = ax.lines[-1]
    # Imposta cap e join arrotondati
    linea_tracciato.set_solid_capstyle('round')   # estremità arrotondate
    linea_tracciato.set_solid_joinstyle('round')  # giunzioni arrotondate

    norm = mpl.colors.Normalize(thr.min(), thr.max())
    lc = LineCollection(segs, cmap='RdYlGn', norm=norm, linewidth=6)
    lc.set_array(thr); lc.set_capstyle('round')
    ax.add_collection(lc)
    cb = plt.colorbar(lc, ax=ax,
                      orientation='horizontal',
                      fraction=0.04, pad=0.05)
    cb.set_label('Throttle (%)', fontsize=8)

    # Estraiamo i punti di frenata e accelerazione per il settore
    brake = tel.loc[mask&(tel['Brake'].astype(int).diff()==1), ['X','Y']] # Punti di frenata, selezionati dalla telemetria del settore e filtrati per il cambiamento del valore del freno e della maschera
    # Punti di accelerazione, dove il valore dell'acceleratore è maggiore di 5% e il valore precedente era minore o uguale a 5%
    accel_mask = (thr>5)&(thr.shift(1)<=5)
    accel = pd.DataFrame({'X':x,'Y':y})[accel_mask] # Estrai i punti di accelerazione dalla telemetria del settore, costruendo un DataFrame con le coordinate X e Y
    # Disegna i punti di frenata e accelerazione sullo stesso asse del grafico
    # ax.scatter(brake['X'], brake['Y'],
    #            marker='v', c='white', s=80,
    #            edgecolors='black', linewidths=1, zorder=3,
    #            label='Brake Point')
    
    for brake_lc in brake_collections: # Aggiungi le collezioni di linee di frenata all'asse
        ax.add_collection(brake_lc)
    
    # Aggiungi numeri dentro i marker
    for i, (brake_coor_x, brake_coor_y) in enumerate(centered_brake_points, start=1):
        ax.text(brake_coor_x, brake_coor_y, str(i), color='black', fontsize=8, fontweight='bold', 
                ha='center', va='center', zorder=4, alpha=0.7)
        ax.scatter(brake_coor_x, brake_coor_y, marker='o', c='white', s=160, 
                   label='Brake Zone '+str(i)+": " + extract_text_from_speed_delta(speed_deltas_for_each_brake_zone[i-1]), edgecolors='black', linewidths=1.5, zorder=3, alpha=0.65)
    # ax.scatter(accel['X'], accel['Y'],
    #            marker='o', c='white', s=80,
    #            edgecolors='black', linewidths=1, zorder=3,
    #            label='Acceleration Point')

    # Stile assi e legenda
    ax.axis('equal'); ax.axis('off')
    ax.legend(loc='best', frameon=False, fontsize=8, markerscale=0.7)

    # Aggiungi una freccia per indicare la direzione del tracciato
    # La freccia è disegnata tra due punti scelti del tracciato
    # Prendi il vettore tangente tra due punti scelti
    dx, dy = x.iloc[3] - x.iloc[1], y.iloc[3] - y.iloc[1]
    L = np.hypot(dx, dy)
    tangent = np.array([dx, dy]) / L

    # Calcola il normale per lo spostamento laterale
    normal = np.array([-dy, dx]) / L
    arrow_offset   = 160            # distanza dalla traccia
    arrow_length   = 250            # lunghezza costante del corpo
    base_idx       = 1              # indice del punto di partenza
    base_point     = np.array([x.iloc[base_idx], y.iloc[base_idx]])

    # Punto di partenza ed end point
    start = base_point + normal * arrow_offset
    end   = start + tangent * arrow_length

    ax.annotate(
        '',               # nessun testo
        xy=end,           # punta
        xytext=start,     # base
        arrowprops=dict(
            arrowstyle='-|>',
            mutation_scale=20,
            linewidth=1.5,
            color='black',
            shrinkA=0,
            shrinkB=0
        ),
        zorder=4,
        annotation_clip=False
    )

    # Inserisci i numeri delle curve del settore
    # Trova i punti della telemetria più vicini alle curve del settore
    sub = corners.loc[sector.index].copy()
    tele_pts = np.column_stack((tel['X'], tel['Y']))
    tree = KDTree(tele_pts)
    coords = np.column_stack((sub['X'], sub['Y']))
    _, idxs = tree.query(coords)
    dxs = tel['X'].iloc[idxs+1].values - tel['X'].iloc[idxs-1].values
    dys = tel['Y'].iloc[idxs+1].values - tel['Y'].iloc[idxs-1].values
    Ls = np.hypot(dxs,dys)
    nxx, nyy = -dys/Ls, dxs/Ls
    sub['X_off'] = sub['X'] + nxx*260
    sub['Y_off'] = sub['Y'] + nyy*260
    for _, r in sub.iterrows():
        ax.text(r['X_off'], r['Y_off'], str(int(r['Number'])),
                fontsize=10, fontweight='bold',
                ha='center', va='center', color='black',
                bbox=dict(facecolor='none', alpha=0.5,
                          edgecolor='none', boxstyle='round,pad=0.2'))

    # Per il lap time, estraiamo il tempo del giro e lo formattiamo usando regex
    lap_time = str(obj['fastest_lap']['LapTime'])[:-3]
    match = re.search(r'(\d+:(\d+:\d+\.\d+))', lap_time)
    if match:
        lap_time = match.group(2)
    # Descrizione di ogni grafico
    ax.set_title(f"{driver} - {weekend.EventName} - {session.event.year} - {ses} - Turn {sector.loc[0,'Number']}\n Compound: {obj['fastest_lap']['Compound']} - LAP: {obj['fastest_lap']['LapNumber'].astype(int)}\n LAP Time: {lap_time}\n Total Laps on this Compound: {obj['stint_length']} Laps",
                 fontsize=8, fontweight='bold')

# --- Sidebar per controlli ---
st.sidebar.title("F1 Telemetry Viewer")
year = st.sidebar.selectbox("Year", [2022, 2023, 2024], index=2)
wknd = st.sidebar.selectbox("Weekend", load_weekends_for_year(year), index=0)
ses = st.sidebar.selectbox("Session", ['FP1', 'FP2', 'FP3', 'Q', 'R'], index=4)
driver = st.sidebar.selectbox("Driver", load_drivers_for_session(year, wknd, ses))

if st.sidebar.button("Load data and show plot"):
# Definisci i parametri dell'analisi

    session = ff1.get_session(year, wknd, ses)
    session.load()
    weekend = session.event

    # Seleziona, per lo specifico pilota, il giro più veloce effettuato per un determinato compound, dopo aver raggruppato i giri per compound
    driver_laps = session.laps.pick_drivers(driver)
    fastest_per_compound = {}
    for comp, grp in driver_laps.groupby('Compound'): # Utilizza il metodo groupby per raggruppare i giri per compound
        stint_per_compund = len(grp) # Estrai i giri per stint
        fastest_lap = grp.pick_fastest() # Estrai il giro più veloce per ogni compound
        if fastest_lap is None: # Se non esiste un giro veloce per il compound, salta
            continue
        fastest_per_compound[comp] = { 'fastest_lap': fastest_lap,
                                    'stint_length': stint_per_compund } # Salva il giro più veloce in un dizionario, per lo specifico compound

    # Estrai le curve del circuito e definisci il settore
    # (curve 1-4 in questo caso)
    corners = session.get_circuit_info().corners
    turns = [1,2,3,4]
    sector = corners[corners['Number'].isin(turns)] # Costruisci il settore delle curve specificate
    dists = corners['Distance'].values
    i0, i1 = sector.index.min(), sector.index.max() # Estrai gli indici della prima e ultima curva del settore
    if i0>0:
        d_prev = dists[i0-1] # d_prev è la distanza dalla curva precedente alla prima curva del settore
    else:
        d_prev = 0 # d_prev è 0 se non esiste una curva precedente, cioè se la curva è la prima del circuito (la distanza è calcolata dalla partenza)
    d0, d_end0 = dists[i0], dists[i1] # d0 è la distanza della prima curva del settore dalla partenza, d_end0 è la distanza dell'ultima curva del settore dalla partenza
    if i1<len(dists)-1:
        d_next = dists[i1+1] # d_next è la distanza della curva successiva all'ultima curva del settore, se esiste
    else:
        d_next = session.laps.get_telemetry()['Distance'].max() # d_next è la distanza massima della telemetria, cioè consideriamo l'ultima curva del circuito
    d_start, d_end = (d_prev+d0)/2, (d_end0+d_next)/2 # Calcola le distanze di inizio e fine del settore



    # Costruisci il grafico per ogni giro del dizionario fastest_per_compound, come un subplot
    n = len(fastest_per_compound)
    cols = n
    rows = 1
    fig, axes = plt.subplots(rows, cols,
                            figsize=(6*cols, 5*rows),
                            tight_layout=True)

    if cols == 1:
        axes = [axes]  # Assicurati che axes sia sempre una lista
    else:
        axes = axes.flatten()
    for ax, obj in zip(axes, fastest_per_compound.values()):
        plot_on_axis(ax, obj)

    # nascondi eventuali subplot vuoti
    for ax in axes[n:]:
        ax.set_visible(False)

    # render in Streamlit
    st.set_page_config(
        page_title="F1 Sector Analysis",
        layout="wide"
    )
    st.pyplot(fig, use_container_width=True)
