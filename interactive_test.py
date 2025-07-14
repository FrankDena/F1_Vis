import streamlit as st
import pandas as pd
import fastf1 as ff1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import KDTree
import io
import base64

# Abilita cache FastF1
ff1.Cache.enable_cache('cache')


def load_weekends_for_year(year):
    schedule = ff1.get_event_schedule(year)
    return schedule['EventName'].tolist()


def load_drivers_for_session(year, wknd, ses):
    session = ff1.get_session(year, wknd, ses)
    session.load()
    driver_numbers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in driver_numbers]
    return drivers


# --- Sidebar ---
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

    turns = [1, 2, 3, 4]
    sector = corners[corners['Number'].isin(turns)]

    dists = corners['Distance'].values
    if not sector.empty:
        idx_first = corners[corners['Number'] == sector.iloc[0]['Number']].index[0]
        idx_last = corners[corners['Number'] == sector.iloc[-1]['Number']].index[0]
    else:
        st.error("Nessun settore trovato con le curve specificate.")
        st.stop()

    d_prev = dists[idx_first-1] if idx_first > 0 else 0
    d0 = dists[idx_first]
    d_end0 = dists[idx_last]
    d_next = dists[idx_last+1] if idx_last < len(dists)-1 else tel['Distance'].max()

    d_start = (d_prev + d0) / 2
    d_end = (d_end0 + d_next) / 2

    sector_telemetry = tel.loc[(tel['Distance'] >= d_start) & (tel['Distance'] <= d_end)].copy()

    x = sector_telemetry['X']
    y = sector_telemetry['Y']
    color = sector_telemetry['Throttle']

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    brake_pts = sector_telemetry.loc[sector_telemetry['Brake'].astype(int).diff() == 1, ['X', 'Y']]
    throttle = sector_telemetry["Throttle"]
    mask_accel_start = (throttle > 5) & (throttle.shift(1) <= 5)
    accel_pts = sector_telemetry.loc[mask_accel_start, ['X', 'Y']]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(f"{driver} - {weekend.EventName} - {session.event.year} - {ses} - Turn {sector.iloc[0]['Number']}",
                 fontsize=14, fontweight='bold')

    # Disegno tracciato nero di sfondo
    ax.plot(x, y, color='black', linewidth=10, alpha=0.8, zorder=1, label='Tracciato')
    linea_tracciato = ax.lines[-1]
    linea_tracciato.set_solid_capstyle('round')
    linea_tracciato.set_solid_joinstyle('round')

    # Segmenti colorati (Acceleratore %)
    norm = mpl.colors.Normalize(color.min(), color.max())
    colormap = mpl.colormaps['RdYlGn']
    lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=6)
    lc.set_capstyle('round')
    lc.set_array(color)
    ax.add_collection(lc)

    # Colorbar
    legend = fig.colorbar(lc, ax=ax, orientation='horizontal', fraction=0.03, pad=0.02)
    legend.set_label('Acceleratore (%)', fontsize=10)

    # Frenata e accelerazione
    ax.scatter(brake_pts['X'], brake_pts['Y'],
               marker='v', c='white', s=120, label='Frenata',
               edgecolors='black', linewidths=1.5, zorder=3)
    ax.scatter(accel_pts['X'], accel_pts['Y'],
               marker='o', c='white', s=120, label='Accelerazione',
               edgecolors='black', linewidths=1.5, zorder=3)

    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axis('off')
    ax.legend(loc='best', frameon=False, fontsize=12)

    # Freccia di direzione
    dx = x.iloc[6] - x.iloc[3]
    dy = y.iloc[6] - y.iloc[3]
    length = np.hypot(dx, dy)
    nx = -dy / length
    ny = dx / length
    offset = 160
    x_offset = nx * offset
    y_offset = ny * offset
    start = (x.iloc[3] + x_offset, y.iloc[3] + y_offset)
    end = (x.iloc[6] + x_offset, y.iloc[6] + y_offset)
    arrow = FancyArrowPatch(posA=start, posB=end, arrowstyle="-|>", mutation_scale=30,
                            color='black', linewidth=2, zorder=4)
    ax.add_patch(arrow)

    # Numeri curva con offset
    curves = corners.loc[idx_first:idx_last]
    telemetry_points = np.column_stack((tel['X'], tel['Y']))
    tree = KDTree(telemetry_points)
    curve_coords = np.column_stack((curves['X'], curves['Y']))
    _, indices = tree.query(curve_coords)
    dx = tel['X'].iloc[indices + 1].values - tel['X'].iloc[indices - 1].values
    dy = tel['Y'].iloc[indices + 1].values - tel['Y'].iloc[indices - 1].values
    lengths = np.hypot(dx, dy)
    nx = -dy / lengths
    ny = dx / lengths
    offset = 260
    curves['X_offset'] = curves['X'] + nx * offset
    curves['Y_offset'] = curves['Y'] + ny * offset
    for _, row in curves.iterrows():
        ax.text(row['X_offset'], row['Y_offset'], f"{row['Number']}",
                fontsize=10, fontweight='bold', color='black',
                ha='center', va='center',
                bbox=dict(facecolor='none', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))

    # Salva figura matplotlib in memoria come PNG (no bbox_inches)
    # Salva figura matplotlib in memoria come PNG (senza bbox_inches)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()

    # Dimensioni reali figura matplotlib
    fig_width, fig_height = fig.get_size_inches()
    fig_dpi = fig.get_dpi()
    width_px = int(fig_width * fig_dpi)
    height_px = int(fig_height * fig_dpi)

    # Bounding box dati per normalizzazione
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Overlay interattivo
    points_json = sector_telemetry[['X', 'Y', 'Throttle']].to_json(orient='records')
    html = f"""
    <div style="position: relative; width: 100%; max-width: {width_px}px;">
        <img id="track" src="data:image/png;base64,{encoded}" style="width: 100%; height: auto;">
        <canvas id="overlay" style="position: absolute; top: 0; left: 0;"></canvas>
    </div>

    <script>
    const img = document.getElementById('track');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const points = {points_json};

    // Ridimensiona canvas per combaciare con l'immagine
    function resizeCanvas() {{
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
    }}
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function drawTooltip(x, y, text) {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.font = "14px Arial";
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        const width = ctx.measureText(text).width + 10;
        ctx.fillRect(x, y - 25, width, 20);
        ctx.fillStyle = "black";
        ctx.fillText(text, x + 5, y - 10);
    }}

    canvas.addEventListener('mousemove', function(event) {{
        const rect = canvas.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;

        let found = false;
        points.forEach(p => {{
            // Mappa coordinate dati → pixel canvas
            const px = (p.X - {x_min}) / ({x_max} - {x_min}) * canvas.width;
            const py = canvas.height - ((p.Y - {y_min}) / ({y_max} - {y_min}) * canvas.height);
            if (Math.abs(mx - px) < 10 && Math.abs(my - py) < 10) {{
                drawTooltip(mx, my, "Acceleratore: " + p.Throttle.toFixed(1) + "%");
                found = true;
            }}
        }});
        if (!found) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }});
    </script>
    """

    st.components.v1.html(html, height=height_px + 50)

