import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== FONT ===== */
html, body, [class*="css"], .stApp, button, input, textarea, select, .stMarkdown {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* =====================================================================
   DESIGN TOKENS
   --------------------------------------------------------------------- */
:root {
    /* COLOR SYSTEM
       ----- Surfaces ------------------------------------------------ */
    --surface:           #FFFFFF;  /* card / panel base */
    --surface-elevated:  #F8FAFC;  /* hover, page bg */
    --surface-elevated2: #F1F5F9;  /* small chips, icon bg */

    /* ----- Borders ------------------------------------------------ */
    --border-subtle:  #E2E8F0;
    --border-strong:  #CBD5E1;
    --border-accent:  #C7D2FE;

    /* ----- Brand & semantic --------------------------------------- */
    --accent:        #4F46E5;             /* primary action */
    --accent-hover:  #4338CA;
    --accent-soft:   rgba(79,70,229,0.08);
    --ok:            #059669;             /* success */
    --warn:          #D97706;             /* attention */
    --danger:        #DC2626;             /* error */
    --locked:        #CBD5E1;             /* disabled */

    /* ----- Text (WCAG AA against --surface-elevated) -------------- */
    --text-primary:   #0F172A;  /* 16.4:1 */
    --text-secondary: #475569;  /*  7.7:1 */
    --text-muted:     #64748B;  /*  5.5:1 — subió desde #94A3B8 (3.5:1) */

    /* ----- Shadows ------------------------------------------------ */
    --shadow-panel:        0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-panel-hover:  0 4px 12px rgba(0,0,0,0.09), 0 1px 3px rgba(0,0,0,0.05);

    /* SPACING SCALE (4-pt) */
    --space-xs:  0.25rem;
    --space-sm:  0.5rem;
    --space-md:  0.75rem;
    --space-lg:  1rem;
    --space-xl:  1.5rem;
    --space-2xl: 2rem;

    /* TYPE SCALE */
    --text-xs:   0.75rem;
    --text-sm:   0.875rem;
    --text-base: 1rem;
    --text-lg:   1.05rem;   /* h3 */
    --text-xl:   1.4rem;    /* h2 */
    --text-2xl:  2rem;      /* h1 / hero */

    /* RADII */
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 10px;
    --radius-xl: 12px;

    /* PANEL */
    --panel-gap:    16px;
    --panel-radius: var(--radius-xl);
    --panel-pad-x:  1.25rem;
    --panel-pad-y:  1.1rem;
}

/* ===== APP BG ===== */
.stApp { background: #F8FAFC !important; }

/* ===== CONTENT AREA ===== */
.block-container {
    max-width: 1440px !important;
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 3rem !important;
}

/* ===== HIDE SIDEBAR ===== */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ===== TYPOGRAPHY ===== */
h1, h2, h3 {
    letter-spacing: -0.025em;
    font-weight: 600;
    color: var(--text-primary);
}
h1 { font-size: var(--text-2xl) !important; margin-bottom: 0.35rem !important; line-height: 1.2; }
h2 { font-size: var(--text-xl) !important; margin-top: 0.9rem !important; margin-bottom: 0.25rem !important; line-height: 1.3; }
h3 { font-size: var(--text-lg) !important; line-height: 1.35; }

/* ===== PANEL BASE ===== */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--panel-radius) !important;
    background: var(--surface) !important;
    padding: var(--panel-pad-y) var(--panel-pad-x) !important;
    box-shadow: var(--shadow-panel) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-panel-hover) !important;
    border-color: #CBD5E1 !important;
}

/* Equal-height panels in horizontal grids */
[data-testid="stHorizontalBlock"] {
    gap: var(--panel-gap) !important;
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 !important;
}

/* ===== PANEL VARIANTS ===== */
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel--hero) {
    background: linear-gradient(140deg, #EEF2FF 0%, #FFFFFF 55%) !important;
    border-color: #C7D2FE !important;
    padding: 1.75rem 2.25rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel--accent) {
    background: #F5F3FF !important;
    border-color: #DDD6FE !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel--ghost) {
    background: transparent !important;
    border-style: dashed !important;
    border-color: #E2E8F0 !important;
    box-shadow: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.panel--success) {
    background: linear-gradient(140deg, #ECFDF5 0%, #FFFFFF 55%) !important;
    border-color: #A7F3D0 !important;
}

/* ===== TOPBAR ===== */
.topbar-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.05rem;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    white-space: nowrap;
    padding: 0.2rem 0;
}
.topbar-brand__mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.5rem; height: 1.5rem;
    border-radius: var(--radius-sm);
    background: linear-gradient(135deg, var(--accent) 0%, #7C3AED 100%);
    color: #FFFFFF;
    font-size: 0.85rem;
    font-weight: 700;
    box-shadow: 0 1px 3px rgba(79,70,229,0.3);
}
.topbar-brand__name { font-weight: 400; color: var(--text-secondary); }
.topbar-brand__name strong { font-weight: 700; color: var(--text-primary); }
.topbar-divider {
    border: none !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    margin: 0.3rem 0 1.5rem 0 !important;
}

/* ===== STEP PILLS ===== */
[data-testid="stVerticalBlock"]:has(.topbar-pill--done):not(:has(.topbar-pill--current)):not(:has(.topbar-pill--locked)) .stButton > button,
[data-testid="stVerticalBlock"]:has(.topbar-pill--current):not(:has(.topbar-pill--done)):not(:has(.topbar-pill--locked)) .stButton > button,
[data-testid="stVerticalBlock"]:has(.topbar-pill--locked):not(:has(.topbar-pill--done)):not(:has(.topbar-pill--current)) .stButton > button {
    border-radius: 999px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: -0.01em !important;
}
/* Done */
[data-testid="stVerticalBlock"]:has(.topbar-pill--done):not(:has(.topbar-pill--current)):not(:has(.topbar-pill--locked)) .stButton > button {
    border-color: #A7F3D0 !important;
    color: #065F46 !important;
    background: #ECFDF5 !important;
}
[data-testid="stVerticalBlock"]:has(.topbar-pill--done):not(:has(.topbar-pill--current)):not(:has(.topbar-pill--locked)) .stButton > button:hover {
    background: #D1FAE5 !important;
}
/* Current */
[data-testid="stVerticalBlock"]:has(.topbar-pill--current):not(:has(.topbar-pill--done)):not(:has(.topbar-pill--locked)) .stButton > button[kind="primary"] {
    box-shadow: 0 1px 3px rgba(79,70,229,.3) !important;
    border-radius: 999px !important;
}

/* ===== BUTTONS ===== */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.875rem;
    border: 1px solid var(--border-subtle);
    transition: all 0.15s ease;
    color: var(--text-secondary);
    background: #FFFFFF;
    letter-spacing: -0.01em;
}
.stButton > button:hover:not(:disabled), .stDownloadButton > button:hover {
    background: #F8FAFC;
    border-color: #CBD5E1;
    color: var(--text-primary);
}
.stButton > button[kind="primary"] {
    background: var(--accent);
    border: 1px solid var(--accent);
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover {
    background: #4338CA;
    border-color: #4338CA;
    box-shadow: 0 4px 12px rgba(79,70,229,.25);
}

/* ===== INPUTS ===== */
textarea, input[type="text"] {
    border-radius: 8px !important;
    border-color: var(--border-subtle) !important;
    font-size: 0.875rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.12) !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: var(--surface);
    padding: 0.85rem 1rem;
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
    box-shadow: var(--shadow-panel);
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted);
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600;
    color: var(--text-primary);
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    font-weight: 500;
    font-size: 0.875rem;
    border-radius: 6px;
    color: var(--text-secondary);
}

/* ===== CAPTIONS ===== */
[data-testid="stCaptionContainer"] {
    color: var(--text-muted);
    line-height: 1.55;
    font-size: 0.8rem;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border-subtle);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.55rem 0.9rem;
    border-radius: 6px 6px 0 0;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
}

/* ===== DIVIDERS ===== */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1.25rem 0 !important;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ===== ALERTS ===== */
[data-testid="stAlert"] { border-radius: 8px; font-size: 0.875rem; }

/* ===== HIDE FOOTER/MENU ===== */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* ===== UTILITY CLASSES ===== */
.panel-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.panel-title {
    font-size: 0.975rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.015em;
}
.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    gap: 0.5rem;
}
.empty-state .empty-icon {
    width: 2.5rem; height: 2.5rem;
    border-radius: 8px;
    background: #F1F5F9;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; font-weight: 700; color: #64748B;
    margin-bottom: 0.5rem;
}
.empty-state .empty-title {
    font-size: 1rem; font-weight: 600; color: var(--text-primary);
}
.empty-state .empty-desc {
    font-size: 0.875rem; color: var(--text-secondary);
    max-width: 320px; line-height: 1.55;
}

/* Dataset chip */
.dataset-chip {
    background: var(--surface-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 0.4rem 0.7rem;
    margin: 0.3rem 0 0;
    font-size: 0.8rem;
    display: inline-block;
}
.dataset-chip .name { font-weight: 500; color: var(--text-primary); margin-bottom: 0.1rem; }
.dataset-chip .meta { color: var(--text-muted); font-size: 0.7rem; }

/* Quality card */
.quality-hero {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 1.1rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: var(--shadow-panel);
}
.qh-grade {
    width: 2.8rem; height: 2.8rem;
    border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem; font-weight: 700;
    flex-shrink: 0;
    border-width: 2px; border-style: solid;
}
.qh-grade--green  { background: #DCFCE7; color: #15803D; border-color: #86EFAC; }
.qh-grade--orange { background: #FEF9C3; color: #B45309; border-color: #FDE68A; }
.qh-grade--red    { background: #FEE2E2; color: #DC2626; border-color: #FCA5A5; }
.qh-grade--gray   { background: #F1F5F9; color: #64748B; border-color: #CBD5E1; }
.qh-score {
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: -0.03em;
    margin-left: 0.5rem;
}
.qh-score--green  { color: #15803D; }
.qh-score--orange { color: #B45309; }
.qh-score--red    { color: #DC2626; }
.qh-score--gray   { color: #64748B; }
.quality-hero .label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--text-muted); margin-bottom: 0.15rem; font-weight: 600;
}
.quality-hero .value { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.15rem; }
.quality-hero .desc { color: var(--text-secondary); font-size: 0.82rem; line-height: 1.4; }

/* Spacing utility classes (use instead of inline style='height:.75rem') */
.space-xs { height: var(--space-xs); }
.space-sm { height: var(--space-sm); }
.space-md { height: var(--space-md); }
.space-lg { height: var(--space-lg); }
.space-xl { height: var(--space-xl); }

/* Custom scrollbar for st.container(height=...) usado en column_selector,
   para que el scroll interno se sienta más profesional */
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stVerticalBlock"]) [data-testid="stVerticalBlock"]::-webkit-scrollbar {
    width: 6px;
}
[data-testid="stVerticalBlock"]::-webkit-scrollbar-track { background: transparent; }
[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
    background: var(--border-strong);
    border-radius: 3px;
}
[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

/* ===================================================================
   WIZARD PROGRESS BAR
   ------------------------------------------------------------------- */
.wizard-progress {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.15rem 0;
}
.wizard-progress__step {
    flex: 1;
    height: 6px;
    background: var(--surface-elevated2);
    border-radius: 3px;
    transition: background 0.25s ease, transform 0.25s ease;
}
.wizard-progress__step--done    { background: var(--ok); }
.wizard-progress__step--current { background: var(--accent); transform: scaleY(1.4); }
.wizard-progress__step--locked  { background: var(--surface-elevated2); }
.wizard-progress__label {
    font-size: var(--text-xs);
    color: var(--text-muted);
    margin-top: 0.25rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 600;
}
.wizard-progress__current-name {
    font-size: var(--text-sm);
    color: var(--text-primary);
    font-weight: 600;
}

/* ===================================================================
   ONBOARDING HERO (paso 1 sin dataset)
   ------------------------------------------------------------------- */
.hero-onboarding {
    text-align: center;
    padding: 1.25rem 0.5rem 0.75rem;
}
.hero-onboarding__mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 3.25rem; height: 3.25rem;
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, var(--accent) 0%, #7C3AED 100%);
    color: #FFFFFF;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 auto 1rem;
    box-shadow: 0 4px 12px rgba(79,70,229,0.25);
}
.hero-onboarding__title {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}
.hero-onboarding__sub {
    font-size: 0.95rem;
    color: var(--text-secondary);
    max-width: 30rem;
    margin: 0 auto 1.5rem;
    line-height: 1.5;
}
.hero-onboarding__hint {
    font-size: var(--text-xs);
    color: var(--text-muted);
    margin-top: 0.6rem;
}

/* ===================================================================
   PIPELINE CHECKLIST (paso 2 mientras corre el análisis)
   ------------------------------------------------------------------- */
.pipeline-checklist { list-style: none; padding: 0; margin: 0; }
.pipeline-checklist li {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0;
    font-size: var(--text-sm);
    color: var(--text-muted);
    transition: color 0.2s ease;
    border-bottom: 1px solid var(--border-subtle);
}
.pipeline-checklist li:last-child { border-bottom: none; }
.pipeline-checklist__marker {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.25rem; height: 1.25rem;
    border-radius: 999px;
    flex-shrink: 0;
    font-size: 0.7rem;
    font-weight: 700;
}
.pipeline-checklist li.is-pending .pipeline-checklist__marker {
    background: var(--surface-elevated2);
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
}
.pipeline-checklist li.is-pending .pipeline-checklist__marker::before { content: ""; width: 4px; height: 4px; background: var(--text-muted); border-radius: 50%; }
.pipeline-checklist li.is-running {
    color: var(--text-primary);
    font-weight: 500;
}
.pipeline-checklist li.is-running .pipeline-checklist__marker {
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid var(--accent);
    animation: pipeline-pulse 1.2s ease-in-out infinite;
}
.pipeline-checklist li.is-running .pipeline-checklist__marker::before { content: ""; width: 6px; height: 6px; background: var(--accent); border-radius: 50%; }
.pipeline-checklist li.is-done {
    color: var(--text-secondary);
}
.pipeline-checklist li.is-done .pipeline-checklist__marker {
    background: var(--ok);
    color: #FFFFFF;
}
.pipeline-checklist li.is-done .pipeline-checklist__marker::before { content: "✓"; }
@keyframes pipeline-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(79,70,229,0.35); }
    50%      { box-shadow: 0 0 0 6px rgba(79,70,229,0); }
}

/* Success hero (post-run) */
.success-hero {
    text-align: center;
    padding: 1.25rem 0.75rem 0.5rem;
}
.success-hero__sparkle {
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
    animation: sparkle-fade 0.6s ease-out both;
}
.success-hero__title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.success-hero__sub {
    color: var(--text-secondary);
    font-size: 0.9rem;
    max-width: 28rem;
    margin: 0 auto 1.25rem;
}
@keyframes sparkle-fade {
    from { opacity: 0; transform: scale(0.7); }
    to   { opacity: 1; transform: scale(1); }
}

/* Archetype cards */
.archetype-card {
    background: var(--surface);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
    box-shadow: var(--shadow-panel);
}
.archetype-card:hover {
    border-color: var(--border-accent);
    box-shadow: 0 4px 12px rgba(79,70,229,.09);
}
.archetype-card .tag {
    display: inline-block;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    font-weight: 600;
}
.archetype-card h3 {
    margin: 0 0 0.5rem 0 !important;
    color: var(--text-primary);
    font-size: 1rem !important;
}
.archetype-card .description {
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 0.85rem;
    font-size: 0.875rem;
}
.archetype-card .prevalence {
    display: inline-block;
    background: var(--accent-soft);
    border: 1px solid #C7D2FE;
    color: #4338CA;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* Importance badge (column selector) */
.importance-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    margin-right: 0.35rem;
}
.importance-badge--high { background: #DCFCE7; color: #15803D; }
.importance-badge--medium { background: #FEF9C3; color: #854D0E; }
.importance-badge--low { background: #F1F5F9; color: #64748B; }

/* Step indicator (legacy compat) */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}
</style>
"""


def inject():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
