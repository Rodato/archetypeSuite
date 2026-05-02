import html
from typing import Dict, List, Optional

import streamlit as st


def render_archetype_cards(
    archetypes: list,
    cluster_sizes: Optional[Dict[int, int]] = None,
):
    if not archetypes:
        st.info("No hay descripciones de arquetipos disponibles.")
        return

    total = sum(cluster_sizes.values()) if cluster_sizes else 0

    for archetype in archetypes:
        cluster_id = archetype.get("cluster_id", "")
        label = html.escape(str(archetype.get("label", "")))
        description = html.escape(str(archetype.get("description", "")))

        prevalence_html = ""
        if cluster_sizes and total > 0:
            try:
                size = cluster_sizes.get(int(cluster_id), 0)
            except (TypeError, ValueError):
                size = 0
            if size > 0:
                pct = (size / total) * 100
                prevalence_html = (
                    f"<div class='prevalence'>👥 {size} personas — "
                    f"{pct:.1f}% de tus datos</div>"
                )

        card_html = [
            "<div class='archetype-card'>",
            f"<div class='tag'>Arquetipo {cluster_id}</div>",
            f"<h3>{label}</h3>",
            prevalence_html,
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
