import streamlit as st
import base64

def load_svg_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

soft = load_svg_as_base64("static/soft.png")
fastest_lap = load_svg_as_base64("static/fastest.png")
lap = load_svg_as_base64("static/lap.png")
logo = load_svg_as_base64("static/f1logo.png")

st.set_page_config(page_title="Lap Analysis Dashboard", layout="wide")

st.image(f"data:image/png;base64,{logo}", width=200)
st.title("F1 Lap Analysis Dashboard")


col1, col2, col3 = st.columns(3, gap="large")


with col1:
   
    st.markdown(
        f"## <img src='data:image/png;base64,{lap}' width='80'> Visualize Single Lap",
        unsafe_allow_html=True
    )
    st.markdown(
        "**Visualize throttle and brake telemetry for the fastest lap of a specific session. Perfect for in-depth analysis of throttle and brake usage by the selected driver.**"
    )
    if st.button("🔎 Show Analysis"):
        st.switch_page("pages/Visualize Single Lap.py")
    



with col2:
    
    
    st.markdown(
        f"## <img src='data:image/png;base64,{fastest_lap}' width='50'> Compare Fastest Laps",
        unsafe_allow_html=True
    )
    st.markdown(
        "**Compare fastest laps of two selected drivers to identify differences in driving style, braking points, and cornering speeds.**"
    )
    if st.button("🔎 Compare Laps"):
        st.switch_page("pages/Compare Fastest Laps.py")
    

with col3:
    
    
    st.markdown(
        f"## <img src='data:image/png;base64,{soft}' width='50'> Compare Compounds",
        unsafe_allow_html=True
    )

    st.write(
        "Compare the fastest lap for each compound used by a driver in the specified session to highlight different approaches."
    )
    if st.button("🔎 Compare Compounds"):
        st.switch_page("pages/Compare Compounds.py")
    

