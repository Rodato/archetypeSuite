import html

import streamlit as st


def render_archetype_cards(archetypes: list):
    if not archetypes:
        st.info("No hay descripciones de arquetipos disponibles.")
        return

    for archetype in archetypes:
        cluster_id = archetype.get("cluster_id", "")
        label = html.escape(str(archetype.get("label", "")))
        description = html.escape(str(archetype.get("description", "")))

        card_html = [
            "<div class='archetype-card'>",
            f"<div class='tag'>Arquetipo {cluster_id}</div>",
            f"<h3>{label}</h3>",
            f"<div class='description'>{description}</div>",
            "</div>",
        ]
        st.markdown("\n".join(card_html), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Características clave**")
            for char in archetype.get("key_characteristics", []) or []:
                st.markdown(f"- {char}")
        with col2:
            st.markdown("**Diferenciadores**")
            for diff in archetype.get("differentiators", []) or []:
                st.markdown(f"- {diff}")

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
