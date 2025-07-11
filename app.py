import streamlit as st
import pandas as pd
import fastf1 as ff1
import plotly.graph_objects as go
from scipy.interpolate import splprep, splev
import numpy as np
from matplotlib import cm, colors

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

    # Filtra un tratto specifico (curve 1-4)
    turns = [1, 2, 3, 4]
    turn = corners[corners['Number'].isin(turns)]

    if not turn.empty:
        dists = corners['Distance'].values
        idx_first = corners[corners['Number'] == turn.loc[0, 'Number']].index[0]
        idx_last = corners[corners['Number'] == turn.loc[len(turn)-1, 'Number']].index[0]

        d_prev = dists[idx_first-1] if idx_first > 0 else 0
        d0 = dists[idx_first]
        d_end0 = dists[idx_last]
        d_next = dists[idx_last+1] if idx_last < len(dists)-1 else tel['Distance'].max()

        d_start = (d_prev + d0) / 2
        d_end = (d_end0 + d_next) / 2

        # Telemetria del tratto
        turn_tel = tel.loc[(tel['Distance'] >= d_start) & (tel['Distance'] <= d_end)].copy()

        # Dati originali
        x_raw = turn_tel['X'].values
        y_raw = turn_tel['Y'].values
        throttle_raw = turn_tel['Throttle'].values

        # Interpolazione spline per un tracciato fluido
        tck, u = splprep([x_raw, y_raw], s=0)
        u_new = np.linspace(0, 1, len(x_raw) * 5)  # Più punti = più fluido
        x_smooth, y_smooth = splev(u_new, tck)
        throttle_smooth = np.interp(u_new, u, throttle_raw)

        # --- Crea tracciato interattivo Plotly ---
        fig = go.Figure()

        # (A) Disegna bordo nero
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            line=dict(color='black', width=10),
            hoverinfo='none',
            showlegend=False
        ))

        # (B) Linea colorata per Throttle
        norm = colors.Normalize(vmin=0, vmax=100)
        cmap = cm.get_cmap('turbo')

        for i in range(len(x_smooth) - 1):
            color = cmap(norm(throttle_smooth[i]))
            hex_color = colors.rgb2hex(color)

            fig.add_trace(go.Scatter(
                x=x_smooth[i:i + 2],
                y=y_smooth[i:i + 2],
                mode='lines',
                line=dict(color=hex_color, width=6),
                hoverinfo='none',
                showlegend=False
            ))

        # (C) Punti frenata e accelerazione
        brk = turn_tel['Brake'].astype(int)
        mask_brake_start = brk.diff() == 1
        brake_pts = turn_tel.loc[mask_brake_start, ['X', 'Y']]

        thr = turn_tel['Throttle']
        mask_accel_start = (thr > 5) & (thr.shift(1) <= 5)
        accel_pts = turn_tel.loc[mask_accel_start, ['X', 'Y']]

        fig.add_trace(go.Scatter(
            x=brake_pts['X'],
            y=brake_pts['Y'],
            mode='markers',
            marker=dict(color='red', size=12, line=dict(color='black', width=1)),
            name='Inizio frenata'
        ))
        fig.add_trace(go.Scatter(
            x=accel_pts['X'],
            y=accel_pts['Y'],
            mode='markers',
            marker=dict(color='green', size=12, line=dict(color='black', width=1)),
            name='Inizio accelerazione'
        ))

        # Layout
        fig.update_layout(
            title=f"{driver} - {weekend.EventName} - {session.event.year} - {ses} - Turn {turn.loc[0,'Number']}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
            height=700,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(x=0.8, y=1)
        )

        # Mostra il grafico
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Nessuna curva trovata nell'intervallo specificato.")
