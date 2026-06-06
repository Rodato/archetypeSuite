import html
from typing import Dict, Optional

import streamlit as st

from src.ui.quality import CAUTION_META

_CAUTION_CLASS = {"green": "qh-grade--green", "orange": "qh-grade--orange", "red": "qh-grade--red"}


def _bullets(title: str, items: list) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for it in items:
        st.markdown(f"- {it}")


def render_archetype_cards(
    archetypes: list,
    cluster_sizes: Optional[Dict[int, int]] = None,
):
    if not archetypes:
        st.info("No hay descripciones de arquetipos disponibles.")
        return

    total = sum(cluster_sizes.values()) if cluster_sizes else 0

    pairs = [archetypes[i:i + 2] for i in range(0, len(archetypes), 2)]

    for pair in pairs:
        cols = st.columns(len(pair), gap="medium")
        for col, archetype in zip(cols, pair):
            with col:
                cluster_id = archetype.get("cluster_id", "")
                label = html.escape(str(archetype.get("label", "")))
                description = html.escape(str(archetype.get("description", "")))

                prevalence_html = ""
                size = 0
                pct = 0.0
                if cluster_sizes and total > 0:
                    try:
                        size = cluster_sizes.get(int(cluster_id), 0)
                    except (TypeError, ValueError):
                        size = 0
                    if size > 0:
                        pct = (size / total) * 100
                        prevalence_html = (
                            f"<div class='prevalence'>{size} personas — "
                            f"{pct:.1f}% del total</div>"
                        )

                caution = archetype.get("nivel_cautela")
                caution_html = ""
                if caution in CAUTION_META:
                    cls = _CAUTION_CLASS.get(CAUTION_META[caution]["color"], "qh-grade--gray")
                    caution_html = (
                        f"<span class='importance-badge {cls}' "
                        f"title='{html.escape(str(archetype.get('cautela_reason', '')))}'>"
                        f"Cautela {html.escape(str(caution))}</span>"
                    )

                card_html = [
                    "<div class='archetype-card'>",
                    f"<div class='tag'>Arquetipo {cluster_id} {caution_html}</div>",
                    f"<h3>{label}</h3>",
                    prevalence_html,
                    f"<div class='description'>{description}</div>",
                    "</div>",
                ]
                st.markdown("\n".join(card_html), unsafe_allow_html=True)

                comportamiento = archetype.get("comportamiento_principal", "")
                micro = archetype.get("microcomportamientos", []) or []
                barreras = archetype.get("barreras", []) or []
                habilitadores = archetype.get("habilitadores", []) or []
                oportunidades = archetype.get("oportunidades_accion", []) or []
                has_behavioral = bool(comportamiento or micro or barreras or habilitadores or oportunidades)

                characteristics = archetype.get("key_characteristics", []) or []
                differentiators = archetype.get("differentiators", []) or []

                if has_behavioral or characteristics or differentiators:
                    with st.expander("Ver detalles"):
                        if has_behavioral:
                            if comportamiento:
                                st.markdown(f"**Comportamiento principal:** {comportamiento}")
                            _bullets("Microcomportamientos", micro)
                            _bullets("Barreras probables", barreras)
                            _bullets("Habilitadores", habilitadores)
                            _bullets("Oportunidades de acción", oportunidades)
                        else:
                            _bullets("Características clave", characteristics)
                            _bullets("Diferenciadores", differentiators)
                        if archetype.get("cautela_reason"):
                            st.caption(archetype["cautela_reason"])

        st.markdown("<div class='space-sm'></div>", unsafe_allow_html=True)
