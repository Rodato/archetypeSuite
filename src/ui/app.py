import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(
    page_title="Archetype Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.sidebar.title("Archetype Suite")
    st.sidebar.markdown("Pipeline agéntico de clustering para generar arquetipos.")

    page = st.sidebar.radio(
        "Navegación",
        ["Cargar Datos", "Perfil de Datos", "Ejecutar Pipeline", "Resultados", "Explorar"],
    )

    if page == "Cargar Datos":
        from src.ui.pages.upload import render
        render()
    elif page == "Perfil de Datos":
        from src.ui.pages.profile import render
        render()
    elif page == "Ejecutar Pipeline":
        from src.ui.pages.pipeline import render
        render()
    elif page == "Resultados":
        from src.ui.pages.results import render
        render()
    elif page == "Explorar":
        from src.ui.pages.explore import render
        render()


if __name__ == "__main__":
    main()
