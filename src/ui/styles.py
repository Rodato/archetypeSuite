import streamlit as st

CUSTOM_CSS = """
<style>
:root {
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(99, 102, 241, 0.4);
    --surface: #1A1D24;
    --surface-elevated: #22262F;
}

.stApp {
    background: linear-gradient(180deg, #0F1115 0%, #141820 100%);
}

/* Tipografía más calmada */
h1, h2, h3 {
    letter-spacing: -0.02em;
    font-weight: 600;
}

h1 { font-size: 2rem !important; margin-bottom: 0.5rem !important; }
h2 { font-size: 1.4rem !important; margin-top: 1.5rem !important; }
h3 { font-size: 1.1rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0B0D11;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.25rem !important;
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #6B7280;
    font-size: 0.8rem;
}

/* Radio del sidebar — más espacio entre pasos */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.45rem 0.25rem;
    border-radius: 6px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.03);
}

/* Contenedores con borde más sutil */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    background: var(--surface);
    padding: 0.5rem 0.25rem;
}

/* Botones */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 500;
    border: 1px solid var(--border-subtle);
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: #6366F1;
    border: 1px solid #6366F1;
}
.stButton > button[kind="primary"]:hover {
    background: #7C7FF5;
    border-color: #7C7FF5;
    transform: translateY(-1px);
}

/* Métricas */
[data-testid="stMetric"] {
    background: var(--surface);
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
}
[data-testid="stMetricLabel"] {
    color: #9CA3AF;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600;
}

/* Expanders */
.streamlit-expanderHeader {
    font-weight: 500;
    border-radius: 6px;
}

/* Caption — más respiración */
[data-testid="stCaptionContainer"] {
    color: #9CA3AF;
    line-height: 1.5;
}

/* Divisor */
hr {
    border-color: var(--border-subtle) !important;
    margin: 2rem 0 !important;
}

/* Tabs más limpios */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    padding: 0.6rem 1rem;
    border-radius: 6px 6px 0 0;
}

/* Esconder el footer "Made with Streamlit" */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* Clases custom */
.quality-hero {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface-elevated) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.quality-hero .emoji {
    font-size: 3rem;
    line-height: 1;
}
.quality-hero .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9CA3AF;
    margin-bottom: 0.25rem;
}
.quality-hero .value {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.quality-hero .desc {
    color: #9CA3AF;
    font-size: 0.9rem;
}

.archetype-card {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.archetype-card:hover {
    border-color: var(--border-accent);
}
.archetype-card .tag {
    display: inline-block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9CA3AF;
    margin-bottom: 0.5rem;
}
.archetype-card h3 {
    margin: 0 0 0.75rem 0 !important;
    color: #F3F4F6;
}
.archetype-card .description {
    color: #D1D5DB;
    line-height: 1.6;
    margin-bottom: 1rem;
}
.archetype-card .prevalence {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #A5B4FC;
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.75rem;
}

.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.95rem;
}

.dataset-chip {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
}
.dataset-chip .name {
    font-weight: 500;
    color: #E5E7EB;
    margin-bottom: 0.15rem;
}
.dataset-chip .meta {
    color: #6B7280;
    font-size: 0.75rem;
}
</style>
"""


def inject():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
