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

# Abilita la cache per velocizzare il caricamento
ff1.Cache.enable_cache('cache')


# --- Funzioni helper ---
def load_weekends_for_year(year):
    """Restituisce la lista degli eventi per un anno specifico."""
    return ff1.get_event_schedule(year)['EventName'].tolist()


def load_drivers_for_session(year, weekend, session_name):
    """Restituisce la lista dei piloti per una sessione specifica."""
    session = ff1.get_session(year, weekend, session_name)
    session.load()
    return [session.get_driver(driver)["Abbreviation"] for driver in session.drivers]


def compute_sector_bounds(corners, sector_curves, telemetry):
    """Calcola i limiti di inizio e fine del settore selezionato."""
    dists = corners['Distance'].values

    idx_first = corners[corners['Number'] == sector_curves.iloc[0]['Number']].index[0]
    idx_last = corners[corners['Number'] == sector_curves.iloc[-1]['Number']].index[0]

    d_prev = dists[idx_first - 1] if idx_first > 0 else 0
    d_next = dists[idx_last + 1] if idx_last < len(dists) - 1 else telemetry['Distance'].max()

    d_start = (d_prev + dists[idx_first]) / 2
    d_end = (dists[idx_last] + d_next) / 2

    return d_start, d_end


def create_colored_track(ax, x, y, color_values):
    """Disegna il tracciato colorato in base all'acceleratore."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = mpl.colors.Normalize(color_values.min(), color_values.max())
    colormap = mpl.colormaps['RdYlGn']
    lc = LineCollection(segments, cmap=colormap, norm=norm, linewidth=6)
    lc.set_array(color_values)
    lc.set_capstyle('round')

    ax.add_collection(lc)

    colorbar = plt.colorbar(lc, ax=ax, orientation='horizontal', fraction=0.03, pad=0.02)
    colorbar.set_label('Throttle (%)', fontsize=10)

    return lc


def add_brake_zones(ax, brake_groups):
    """Disegna le zone di frenata sul tracciato."""
    for _, group in brake_groups:
        xg, yg = group['X'].values, group['Y'].values
        points = np.array([xg, yg]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        brake_lc = LineCollection(segments, colors="white", linewidths=3, linestyle='-')
        brake_lc.set_capstyle('round')
        ax.add_collection(brake_lc)


def annotate_brake_points(ax, brake_groups):
    """Aggiunge marker e testi per le zone di frenata."""
    for i, (_, group) in enumerate(brake_groups, start=1):
        center_idx = len(group) // 3
        x_center, y_center = group.iloc[center_idx][['X', 'Y']]

        speeds = group['Speed'].values
        delta = int(speeds[0]) - int(speeds[-1]) if len(speeds) > 1 else 0

        text = f"{int(speeds[0])}→{int(speeds[-1])} km/h (Δ={delta} km/h)"
        ax.scatter(x_center, y_center, marker='o', c='white', s=160,
                   edgecolors='black', linewidths=1.5, zorder=3, alpha=0.65,
                   label=f'Brake Zone {i}: {text}')
        ax.text(x_center, y_center, str(i), color='black', fontsize=8,
                ha='center', va='center', zorder=4, alpha=0.7)


def annotate_corners(ax, corners, telemetry):
    """Annota i numeri delle curve fuori dal tracciato."""
    telemetry_points = np.column_stack((telemetry['X'], telemetry['Y']))
    tree = KDTree(telemetry_points)
    _, indices = tree.query(np.column_stack((corners['X'], corners['Y'])))

    dx = telemetry['X'].iloc[indices + 1].values - telemetry['X'].iloc[indices - 1].values
    dy = telemetry['Y'].iloc[indices + 1].values - telemetry['Y'].iloc[indices - 1].values
    lengths = np.hypot(dx, dy)
    nx, ny = -dy / lengths, dx / lengths

    offset = 260
    x_offsets = corners['X'] + nx * offset
    y_offsets = corners['Y'] + ny * offset

    for x, y, num in zip(x_offsets, y_offsets, corners['Number']):
        ax.text(x, y, str(num), fontsize=10, fontweight='bold',
                color='black', ha='center', va='center',
                bbox=dict(facecolor='none', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))


def draw_direction_arrow(ax, x, y):
    """Disegna una freccia per indicare la direzione del tracciato."""
    dx, dy = x.iloc[6] - x.iloc[3], y.iloc[6] - y.iloc[3]
    length = np.hypot(dx, dy)
    nx, ny = -dy / length, dx / length

    offset = 160
    start = (x.iloc[3] + nx * offset, y.iloc[3] + ny * offset)
    end = (x.iloc[6] + nx * offset, y.iloc[6] + ny * offset)

    arrow = FancyArrowPatch(posA=start, posB=end,
                            arrowstyle="-|>", mutation_scale=30,
                            color='black', linewidth=2, zorder=4)
    ax.add_patch(arrow)


def format_lap_time(lap_time):
    """Formatta il tempo del giro."""
    lap_time_str = str(lap_time)[:-3]
    match = re.search(r'(\d+:(\d+:\d+\.\d+))', lap_time_str)
    return match.group(2) if match else lap_time_str


# --- Streamlit UI ---
st.sidebar.title("F1 Telemetry Viewer")
year = st.sidebar.selectbox("Year", [2022, 2023, 2024], index=2)
weekend = st.sidebar.selectbox("Weekend", load_weekends_for_year(year))
session_name = st.sidebar.selectbox("Session", ['FP1', 'FP2', 'FP3', 'Q', 'R'], index=4)
drivers = load_drivers_for_session(year, weekend, session_name)
driver = st.sidebar.selectbox("Driver", drivers)

if st.sidebar.button("Load data and show plot"):
    session = ff1.get_session(year, weekend, session_name)
    session.load()
    lap = session.laps.pick_drivers(driver).pick_fastest()
    telemetry = lap.get_telemetry()
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners

    sector_curves = corners[corners['Number'].isin([1, 2, 3, 4])]
    if sector_curves.empty:
        st.error("Nessun settore trovato con le curve specificate.")
        st.stop()

    d_start, d_end = compute_sector_bounds(corners, sector_curves, telemetry)
    sector_telemetry = telemetry[(telemetry['Distance'] >= d_start) & (telemetry['Distance'] <= d_end)].copy()

    x, y, throttle = sector_telemetry['X'], sector_telemetry['Y'], sector_telemetry['Throttle']
    brake = sector_telemetry['Brake']
    brake_groups = sector_telemetry[brake == 1].groupby((brake != brake.shift()).cumsum())

    fig, ax = plt.subplots(figsize=(6, 6))
    lap_time_formatted = format_lap_time(lap['LapTime'])
    fig.suptitle(
        f"{driver} - {session.event.EventName} {session.event.year} {session_name} "
        f"Sector {sector_curves.iloc[0]['Number']}\n"
        f"Compound: {lap['Compound']} - Lap: {int(lap['LapNumber'])} - Time: {lap_time_formatted}",
        fontsize=12, fontweight='bold'
    )

    # Disegno tracciato
    ax.plot(x, y, color='black', linewidth=10, alpha=0.8, zorder=1)
    ax.lines[-1].set_solid_capstyle('round')
    ax.lines[-1].set_solid_joinstyle('round')

    create_colored_track(ax, x, y, throttle)
    add_brake_zones(ax, brake_groups)
    annotate_brake_points(ax, brake_groups)
    annotate_corners(ax, sector_curves, telemetry)
    draw_direction_arrow(ax, x, y)

    ax.axis('equal')
    ax.axis('off')
    ax.legend(loc='best', frameon=False, fontsize=7, markerscale=0.5)

    st.pyplot(fig)
