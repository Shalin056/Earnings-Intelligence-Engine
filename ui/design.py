"""
ECRIE Design System
===================
All CSS, color tokens, and HTML component builders live here.
app.py imports from this file — zero CSS in app.py.

Aesthetic: Dark financial terminal. Deep navy base, sharp teal accents,
gold for warnings/dev-mode. Monospace data feel meets modern dashboard.
"""

# ── Master CSS injected once at startup ────────────────────────────────
MASTER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
  --navy-deep:   #080E1C;
  --navy-mid:    #0F1829;
  --navy-card:   #162034;
  --navy-border: #1E2E48;
  --teal:        #00C9A7;
  --teal-dark:   #009E83;
  --gold:        #F0A500;
  --red:         #F05454;
  --green:       #30D158;
  --blue:        #4D9FFF;
  --text-hi:     #EEF2F8;
  --text-mid:    #8FA3BF;
  --text-lo:     #3D5470;
  --font-ui:     'Space Grotesk', sans-serif;
  --font-mono:   'JetBrains Mono', monospace;
}

.stApp { background: var(--navy-deep) !important; color: var(--text-hi) !important; font-family: var(--font-ui) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] { background: var(--navy-mid) !important; border-right: 1px solid var(--navy-border) !important; }
[data-testid="stSidebar"] * { color: var(--text-hi) !important; font-family: var(--font-ui) !important; }
[data-testid="stSidebar"] .stButton > button {
  border: none !important; background: transparent !important;
  color: var(--text-mid) !important; text-align: left !important;
  font-size: 0.82rem !important; padding: 0.45rem 0.8rem !important;
  border-radius: 4px !important; width: 100% !important;
  margin-bottom: 2px !important; letter-spacing: normal !important; text-transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(0,201,167,0.1) !important; color: var(--teal) !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--navy-deep); }
::-webkit-scrollbar-thumb { background: var(--navy-border); border-radius: 2px; }

.stButton > button {
  background: transparent !important; color: var(--teal) !important;
  border: 1px solid var(--teal) !important; border-radius: 4px !important;
  font-family: var(--font-ui) !important; font-size: 0.8rem !important; transition: all 0.18s !important;
}
.stButton > button:hover { background: var(--teal) !important; color: var(--navy-deep) !important; }
.stButton > button[kind="primary"] { background: var(--teal) !important; color: var(--navy-deep) !important; font-weight: 600 !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: var(--teal-dark) !important; }

.stTextInput input, .stSelectbox > div > div, .stNumberInput input, .stTextArea textarea {
  background: var(--navy-card) !important; border: 1px solid var(--navy-border) !important;
  color: var(--text-hi) !important; border-radius: 4px !important;
  font-family: var(--font-mono) !important; font-size: 0.82rem !important;
}

.stWarning { background: rgba(240,165,0,0.08) !important; border: 1px solid rgba(240,165,0,0.25) !important; border-radius: 4px !important; }
.stSuccess { background: rgba(48,209,88,0.08) !important; border: 1px solid rgba(48,209,88,0.25) !important; border-radius: 4px !important; }
.stInfo    { background: rgba(77,159,255,0.08) !important; border: 1px solid rgba(77,159,255,0.25) !important; border-radius: 4px !important; }
.stError   { background: rgba(240,84,84,0.08) !important;  border: 1px solid rgba(240,84,84,0.25) !important;  border-radius: 4px !important; }

[data-testid="stDataFrameContainer"] { border: 1px solid var(--navy-border) !important; border-radius: 6px !important; }

.stTabs [data-baseweb="tab"] { font-family: var(--font-ui) !important; font-size: 0.82rem !important; color: var(--text-mid) !important; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; border-bottom-color: var(--teal) !important; }

.streamlit-expanderHeader { background: var(--navy-card) !important; border: 1px solid var(--navy-border) !important; border-radius: 4px !important; font-family: var(--font-ui) !important; font-size: 0.82rem !important; color: var(--text-hi) !important; }

.stProgress .st-bo { background: var(--teal) !important; }

hr { border-color: var(--navy-border) !important; margin: 1.5rem 0 !important; }

[data-testid="stMetric"] { background: var(--navy-card); border: 1px solid var(--navy-border); border-radius: 6px; padding: 1rem; }
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════
# HTML COMPONENT BUILDERS
# ══════════════════════════════════════════════════════════════════════

def top_bar(title="ECRIE", subtitle="Earnings Call & Risk Intelligence Engine",
            badge="", badge_color="#00C9A7"):
    badge_html = (
        f'<span style="background:{badge_color};color:#080E1C;font-size:0.62rem;'
        f'font-weight:700;padding:0.25rem 0.7rem;border-radius:3px;'
        f'letter-spacing:0.1em;text-transform:uppercase;">{badge}</span>'
    ) if badge else ""
    return f"""
    <div style="background:linear-gradient(135deg,#0F1829 0%,#080E1C 100%);
                border-bottom:2px solid #00C9A7;padding:1.2rem 1.5rem;
                margin:-1rem -1rem 2rem -1rem;
                display:flex;align-items:center;justify-content:space-between;">
      <div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;
                    font-weight:700;color:#00C9A7;letter-spacing:0.04em;">ECRIE</div>
        <div style="font-size:0.62rem;color:#3D5470;letter-spacing:0.14em;
                    text-transform:uppercase;margin-top:0.15rem;">{subtitle}</div>
      </div>
      {badge_html}
    </div>"""


def kpi(label, value, note="", accent="#00C9A7"):
    return f"""
    <div style="background:#162034;border:1px solid #1E2E48;border-top:2px solid {accent};
                border-radius:6px;padding:1.1rem 1.2rem;overflow:hidden;">
      <div style="font-size:0.62rem;color:#3D5470;letter-spacing:0.14em;
                  text-transform:uppercase;margin-bottom:0.45rem;">{label}</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.9rem;
                  font-weight:700;color:{accent};line-height:1;margin-bottom:0.25rem;">{value}</div>
      <div style="font-size:0.68rem;color:#8FA3BF;">{note}</div>
    </div>"""


def card(title, body_html, accent="#1E2E48"):
    return f"""
    <div style="background:#162034;border:1px solid {accent};border-radius:6px;
                padding:1.3rem;margin-bottom:1rem;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:0.88rem;font-weight:600;
                  color:#EEF2F8;margin-bottom:1rem;padding-bottom:0.6rem;
                  border-bottom:1px solid #1E2E48;">{title}</div>
      {body_html}
    </div>"""


def agent_entry(agent, decision, rationale, timestamp="", style="teal"):
    colors = {"teal": "#00C9A7", "gold": "#F0A500", "red": "#F05454", "green": "#30D158"}
    c = colors.get(style, "#00C9A7")
    ts = (f'<div style="font-size:0.6rem;color:#3D5470;margin-top:0.35rem;">'
          f'{timestamp[:19]}</div>') if timestamp else ""
    return f"""
    <div style="background:#0F1829;border:1px solid #1E2E48;border-left:3px solid {c};
                padding:0.8rem 1rem;margin-bottom:0.5rem;border-radius:3px;">
      <div style="font-size:0.6rem;color:{c};letter-spacing:0.12em;
                  text-transform:uppercase;margin-bottom:0.2rem;">{agent}</div>
      <div style="font-size:0.8rem;color:#EEF2F8;font-weight:500;
                  margin-bottom:0.18rem;">{decision}</div>
      <div style="font-size:0.72rem;color:#8FA3BF;">↳ {rationale}</div>
      {ts}
    </div>"""


def pill(text, color="teal"):
    cfg = {
        "teal":  ("rgba(0,201,167,0.12)",  "#00C9A7", "rgba(0,201,167,0.3)"),
        "gold":  ("rgba(240,165,0,0.12)",  "#F0A500", "rgba(240,165,0,0.3)"),
        "red":   ("rgba(240,84,84,0.12)",  "#F05454", "rgba(240,84,84,0.3)"),
        "green": ("rgba(48,209,88,0.12)",  "#30D158", "rgba(48,209,88,0.3)"),
        "blue":  ("rgba(77,159,255,0.12)", "#4D9FFF", "rgba(77,159,255,0.3)"),
    }
    bg, fg, border = cfg.get(color, cfg["teal"])
    return (f'<span style="background:{bg};color:{fg};border:1px solid {border};'
            f'font-size:0.62rem;font-weight:700;padding:0.2rem 0.6rem;'
            f'border-radius:3px;letter-spacing:0.1em;text-transform:uppercase;">{text}</span>')


def section_head(title, sub=""):
    sub_html = (f'<div style="font-size:0.62rem;color:#3D5470;letter-spacing:0.14em;'
                f'text-transform:uppercase;margin-top:0.2rem;">{sub}</div>') if sub else ""
    return f"""
    <div style="margin-bottom:1.2rem;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                  font-weight:700;color:#EEF2F8;">{title}</div>
      {sub_html}
    </div>"""


def signal_display(signal, prob, conviction):
    cfg = {"BUY": ("#30D158", "▲"), "SELL": ("#F05454", "▼"), "HOLD": ("#F0A500", "■")}
    color, icon = cfg.get(signal.upper(), ("#8FA3BF", "●"))
    return f"""
    <div style="background:#162034;border:2px solid {color};border-radius:8px;
                padding:2rem;text-align:center;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:3rem;font-weight:700;
                  color:{color};line-height:1;margin-bottom:0.5rem;">{icon} {signal.upper()}</div>
      <div style="font-size:0.68rem;color:#8FA3BF;letter-spacing:0.14em;text-transform:uppercase;">
        {prob:.0%} model probability &nbsp;·&nbsp; {conviction} conviction
      </div>
    </div>"""


def dev_warn():
    return """
    <div style="background:rgba(240,165,0,0.08);border:1px solid rgba(240,165,0,0.3);
                border-radius:4px;padding:0.6rem 1rem;margin-bottom:1.2rem;
                font-size:0.72rem;color:#F0A500;">
      ⚠️ &nbsp;<strong>DEVELOPER MODE</strong> — Pipeline internals visible.
      Do not share this session with end-users.
    </div>"""


def footer():
    return """
    <div style="text-align:center;padding:2rem 0 1rem;font-size:0.62rem;color:#3D5470;
                letter-spacing:0.12em;text-transform:uppercase;
                border-top:1px solid #1E2E48;margin-top:3rem;">
      ECRIE &nbsp;·&nbsp; FSE 570 Capstone &nbsp;·&nbsp;
      Arizona State University &nbsp;·&nbsp; Spring 2026
    </div>"""


def plotly_theme():
    """Dark Plotly layout. Use: fig.update_layout(**plotly_theme())"""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0F1829",
        font=dict(family="Space Grotesk", color="#8FA3BF", size=11),
        title_font=dict(family="Space Grotesk", color="#EEF2F8", size=13),
        xaxis=dict(gridcolor="#1E2E48", linecolor="#1E2E48", tickcolor="#3D5470"),
        yaxis=dict(gridcolor="#1E2E48", linecolor="#1E2E48", tickcolor="#3D5470"),
        legend=dict(bgcolor="rgba(22,32,52,0.9)", bordercolor="#1E2E48", borderwidth=1),
        colorway=["#00C9A7", "#F0A500", "#4D9FFF", "#F05454", "#30D158", "#9B72CF"],
        margin=dict(t=40, b=20, l=10, r=10),
    )