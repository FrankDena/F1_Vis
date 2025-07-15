import streamlit as st

st.set_page_config(page_title="Lap Analysis Dashboard", layout="wide")

st.title("🏁 F1 Lap Analysis")
st.subheader("📊 Choose an option:")

# Colonne per le due card
col1, col2 = st.columns(2, gap="large")

# 🎨 Stile CSS per rendere le card più eleganti
card_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #f0f0f0;
        color: #333333;
        height: 150px;
        width: 100%;
        border: 2px solid #cccccc;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff4b4b;
        color: white;
        border-color: #ff4b4b;
    }
    </style>
"""
st.markdown(card_style, unsafe_allow_html=True)

# 🚀 Card 1: Fastest lap plot
with col1:
    if st.button("🚀 Show fastest lap throttle and brake plot"):
        st.success("Loading the throttle/brake plot of the fastest lap...")
        # 👇 Qui puoi richiamare la tua funzione
        # show_fastest_lap_plot()

# 📊 Card 2: Compare laps
with col2:
    if st.button("📊 Compare laps"):
        st.success("Loading the throttle/brake plot comparison of multiple laps...")
        # 👇 Qui puoi richiamare la tua funzione
        # compare_laps()
