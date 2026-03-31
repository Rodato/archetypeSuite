import streamlit as st


def render_archetype_cards(archetypes: list):
    if not archetypes:
        st.info("No hay descripciones de arquetipos disponibles.")
        return

    for archetype in archetypes:
        with st.container(border=True):
            st.subheader(f"Cluster {archetype['cluster_id']}: {archetype['label']}")
            st.markdown(archetype["description"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Características Clave**")
                for char in archetype.get("key_characteristics", []):
                    st.markdown(f"- {char}")
            with col2:
                st.markdown("**Diferenciadores**")
                for diff in archetype.get("differentiators", []):
                    st.markdown(f"- {diff}")
