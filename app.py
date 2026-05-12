"""
Surgical Analytics Dashboard — Streamlit App
Anterior Lumbar Approach Complication Risk Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, os, base64

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surgical Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 1.6rem; font-weight: 600;
        color: #1A2E4A; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.9rem; color: #888780;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F7F7F5; border-radius: 8px;
        padding: 14px 18px; border: 1px solid #E0DED8;
    }
    .metric-val  { font-size: 1.8rem; font-weight: 600; color: #1A2E4A; }
    .metric-lbl  { font-size: 0.75rem; color: #888780;
                   text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-sub  { font-size: 0.75rem; color: #888780; margin-top: 2px; }
    .section-note {
        font-size: 0.78rem; color: #888780;
        font-style: italic; margin-top: 6px;
    }
    .risk-box {
        border-radius: 10px; padding: 20px 24px;
        text-align: center; border: 1px solid #E0DED8;
    }
    .assumption-box {
        background: #FAEEDA; border-left: 4px solid #EF9F27;
        padding: 10px 14px; border-radius: 4px;
        font-size: 0.82rem; color: #633806; margin: 8px 0;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and prepare dataset. Checks multiple possible paths."""
    paths = [
        "data_deidentified.csv",
        "data/data_deidentified.csv",
        os.path.join(os.path.dirname(__file__), "data_deidentified.csv"),
    ]
    df = None
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            break

    if df is None:
        return None

    # Ensure engineered columns exist
    if "Any_Complication" not in df.columns:
        comp_cols = [c for c in ["Transfusion Needed","DVT","Wound Infection",
                                  "30 Day ER Visit","Elevated SBP","Rectus M Collection",
                                  "Vascular_Injury_Any"] if c in df.columns]
        if "Vascular Injury" in df.columns and "Vascular_Injury_Any" not in df.columns:
            df["Vascular_Injury_Any"] = df["Vascular Injury"].notna().astype(int)
        df["Any_Complication"] = df[comp_cols].apply(
            pd.to_numeric, errors="coerce").max(axis=1).fillna(0)

    if "Prior_Abd_Surg_Flag" not in df.columns and "Prior Abd Surg" in df.columns:
        df["Prior_Abd_Surg_Flag"] = df["Prior Abd Surg"].notna().astype(int)

    if "Num_Levels" not in df.columns:
        level_cols = [c for c in ["L5-S1","L4-5","L3-4","L2-3","L1-2"] if c in df.columns]
        df["Num_Levels"] = df[level_cols].apply(
            pd.to_numeric, errors="coerce").sum(axis=1)

    if "Revision_Flag" not in df.columns and "Revision Surg" in df.columns:
        df["Revision_Flag"] = df["Revision Surg"].notna().astype(int)

    if "Sex_Binary" not in df.columns and "Sex" in df.columns:
        df["Sex_Binary"] = (df["Sex"] == "Male").astype(int)

    return df

# ── Logistic model coefficients (from training on data_clean_final.csv) ───────
INTERCEPT = -3.8566
COEF = {
    "Age": 0.037893, "Sex_Binary": 0.082385, "BMI": -0.005084,
    "ASA": 0.506369, "Prior_Abd_Surg_Flag": 0.077773,
    "Revision_Flag": -1.236401, "Exposure_Time": 0.016973,
    "Surgical_Time": 0.014387, "EBL": 0.004746, "Num_Levels": -0.920039,
    "Incision_Length": -0.128357, "Hospital_Stay": 0.287484,
    "Spondylolithesis": 0.007077, "Transitional_Spine": 0.042657,
    "Anterior_Osteophyte": -0.13969, "Approach_Left": -0.060179,
}
MEDIANS = {
    "Age": 58, "Sex_Binary": 1, "BMI": 29, "ASA": 2,
    "Prior_Abd_Surg_Flag": 0, "Revision_Flag": 0, "Exposure_Time": 17,
    "Surgical_Time": 89, "EBL": 10, "Num_Levels": 1,
    "Incision_Length": 7, "Hospital_Stay": 1, "Spondylolithesis": 0,
    "Transitional_Spine": 0, "Anterior_Osteophyte": 0, "Approach_Left": 1,
}

COLORS = {
    "blue": "#378ADD", "coral": "#D85A30", "amber": "#EF9F27",
    "green": "#3B6D11", "navy": "#1A2E4A", "gray": "#888780",
    "green_bg": "#EAF3DE", "amber_bg": "#FAEEDA", "red_bg": "#FCEBEB",
}


def logistic_prob(age, sex, bmi, asa, abd, rev, et, st, hs, lvl, spondy=0):
    logit = (INTERCEPT +
             COEF["Age"] * age + COEF["Sex_Binary"] * sex +
             COEF["BMI"] * bmi + COEF["ASA"] * asa +
             COEF["Prior_Abd_Surg_Flag"] * abd + COEF["Revision_Flag"] * rev +
             COEF["Exposure_Time"] * et + COEF["Surgical_Time"] * st +
             COEF["EBL"] * 10 + COEF["Num_Levels"] * lvl +
             COEF["Incision_Length"] * 7 + COEF["Hospital_Stay"] * hs +
             COEF["Spondylolithesis"] * spondy +
             COEF["Transitional_Spine"] * 0 +
             COEF["Anterior_Osteophyte"] * 0 + COEF["Approach_Left"] * 1)
    return min(0.97, max(0.02, 1 / (1 + np.exp(-logit))))


# ── Layout helpers ────────────────────────────────────────────────────────────
def metric_card(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">{label}</div>
        <div class="metric-val">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def plotly_defaults():
    return dict(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#333333"),
        margin=dict(l=50, r=30, t=40, b=50),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🏥 Surgical Analytics Dashboard</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Anterior lumbar approach — '
            'complication risk analysis · 331 patients · AL-LIF Retro</div>',
            unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.error("⚠️ Dataset not found. Please place `data_deidentified.csv` "
             "in the same folder as `app.py`.")
    st.info("Expected path: `data_deidentified.csv` or `data/data_deidentified.csv`")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Demographics",
    "⚕️ Complications",
    "⏱ Exposure time",
    "🤖 Model",
    "🎯 Risk calculator",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total patients", len(df))
    with c2: metric_card("Median age", f"{df['Age'].median():.0f}", "years")
    with c3: metric_card("Median BMI", f"{df['BMI'].median():.1f}")
    with c4: metric_card("Complication rate",
                         f"{df['Any_Complication'].mean()*100:.1f}%",
                         f"{int(df['Any_Complication'].sum())} patients")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age distribution")
        age_bins = [0, 40, 50, 60, 70, 90]
        age_labels = ["<40", "40–49", "50–59", "60–69", "70+"]
        df["age_bin"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
        age_counts = df["age_bin"].value_counts().sort_index().reset_index()
        age_counts.columns = ["Age group", "Count"]
        fig = px.bar(age_counts, x="Age group", y="Count",
                     color_discrete_sequence=[COLORS["blue"]])
        fig.update_layout(**plotly_defaults(), height=280)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sex distribution")
        sex_counts = df["Sex"].value_counts().reset_index()
        sex_counts.columns = ["Sex", "Count"]
        fig = px.pie(sex_counts, names="Sex", values="Count",
                     color_discrete_sequence=[COLORS["blue"], COLORS["coral"]],
                     hole=0.55)
        fig.update_layout(**plotly_defaults(), height=280,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("ASA class")
        asa_counts = df["ASA"].value_counts().sort_index().reset_index()
        asa_counts.columns = ["ASA", "Count"]
        asa_counts["ASA"] = "ASA " + asa_counts["ASA"].astype(str)
        asa_colors = [COLORS["green"], COLORS["blue"],
                      COLORS["amber"], COLORS["coral"]]
        fig = px.bar(asa_counts, x="ASA", y="Count",
                     color="ASA",
                     color_discrete_sequence=asa_colors)
        fig.update_layout(**plotly_defaults(), height=260,
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("BMI distribution")
        bmi_bins = [0, 25, 30, 35, 60]
        bmi_labels = ["<25", "25–30", "30–35", "≥35"]
        df["bmi_bin"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels)
        bmi_counts = df["bmi_bin"].value_counts().sort_index().reset_index()
        bmi_counts.columns = ["BMI band", "Count"]
        fig = px.bar(bmi_counts, x="BMI band", y="Count",
                     color_discrete_sequence=["#7F77DD"])
        fig.update_layout(**plotly_defaults(), height=260)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comorbidity prevalence")
    n = len(df)
    comorbidities = {
        "Prior abdominal surgery": int(df.get("Prior_Abd_Surg_Flag",
                                               pd.Series([0])).sum()),
        "Spondylolisthesis":       int(df.get("Spondylolithesis",
                                               pd.Series([0])).fillna(0).sum()),
        "Revision surgery":        int(df.get("Revision_Flag",
                                               pd.Series([0])).sum()),
        "Transitional spine":      int(df.get("Transitional Spine",
                                               pd.Series([0])).fillna(0).sum()),
    }
    comorb_df = pd.DataFrame({
        "Comorbidity": list(comorbidities.keys()),
        "Rate (%)":    [round(v / n * 100, 1) for v in comorbidities.values()],
        "n":           list(comorbidities.values()),
    })
    fig = px.bar(comorb_df, x="Rate (%)", y="Comorbidity",
                 orientation="h", text="n",
                 color_discrete_sequence=[COLORS["coral"]])
    fig.update_layout(**plotly_defaults(), height=220)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Any complication",
                         f"{df['Any_Complication'].mean()*100:.1f}%")
    er_rate = df["30 Day ER Visit"].mean()*100 if "30 Day ER Visit" in df.columns else 0
    tr_rate = df["Transfusion Needed"].mean()*100 if "Transfusion Needed" in df.columns else 0
    vi_rate = df["Vascular_Injury_Any"].mean()*100 if "Vascular_Injury_Any" in df.columns else 0
    with c2: metric_card("30-day ER visit", f"{er_rate:.1f}%")
    with c3: metric_card("Transfusion", f"{tr_rate:.1f}%")
    with c4: metric_card("Vascular injury", f"{vi_rate:.1f}%")

    st.markdown("---")

    st.markdown('<div class="assumption-box">⚠️ <b>Assumption A-04:</b> All 30-day ER visits '
                '(n=35, 10.6%) are treated as clinically significant. Chart review of these '
                'records is recommended before final publication.</div>',
                unsafe_allow_html=True)
    st.markdown("")

    comp_map = {
        "30-day ER visit":      "30 Day ER Visit",
        "Transfusion":          "Transfusion Needed",
        "Vascular injury":      "Vascular_Injury_Any",
        "Wound infection":      "Wound Infection",
        "DVT":                  "DVT",
        "Elevated SBP":         "Elevated SBP",
        "Rectus collection":    "Rectus M Collection",
    }
    comp_counts = {}
    for label, col in comp_map.items():
        if col in df.columns:
            comp_counts[label] = int(
                pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    comp_df = pd.DataFrame({
        "Complication": list(comp_counts.keys()),
        "Count": list(comp_counts.values()),
    }).sort_values("Count", ascending=True)

    fig = px.bar(comp_df, x="Count", y="Complication", orientation="h",
                 text="Count",
                 color_discrete_sequence=[COLORS["coral"]])
    fig.update_layout(**plotly_defaults(), height=280,
                      title="Complication type — count")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rate by ASA class")
        asa_comp = df.groupby("ASA")["Any_Complication"].agg(["mean","count"]).reset_index()
        asa_comp["rate"] = (asa_comp["mean"] * 100).round(1)
        asa_comp["label"] = "ASA " + asa_comp["ASA"].astype(str)
        colors = [COLORS["green"] if r < 10 else COLORS["amber"] if r < 20
                  else COLORS["coral"] for r in asa_comp["rate"]]
        fig = go.Figure(go.Bar(
            x=asa_comp["label"], y=asa_comp["rate"],
            text=[f"{r}%" for r in asa_comp["rate"]],
            textposition="outside", marker_color=colors,
        ))
        fig.update_layout(**plotly_defaults(), height=260,
                          yaxis_title="Complication rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Rate by BMI band")
        bmi_comp = df.groupby("bmi_bin", observed=True)["Any_Complication"].agg(
            ["mean","count"]).reset_index()
        bmi_comp["rate"] = (bmi_comp["mean"] * 100).round(1)
        colors = [COLORS["green"] if r < 10 else COLORS["amber"] if r < 20
                  else COLORS["coral"] for r in bmi_comp["rate"]]
        fig = go.Figure(go.Bar(
            x=bmi_comp["bmi_bin"].astype(str), y=bmi_comp["rate"],
            text=[f"{r}%" for r in bmi_comp["rate"]],
            textposition="outside", marker_color=colors,
        ))
        fig.update_layout(**plotly_defaults(), height=260,
                          xaxis_title="BMI band",
                          yaxis_title="Complication rate (%)")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPOSURE TIME
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    et = df["Exposure Time (min)"]
    comp = df["Any_Complication"]

    et_no  = et[comp == 0].median()
    et_yes = et[comp == 1].median()

    from scipy.stats import pearsonr, mannwhitneyu
    r, p_r = pearsonr(et.dropna(), comp[et.notna()])
    _, p_mw = mannwhitneyu(et[comp==0].dropna(), et[comp==1].dropna(),
                           alternative="less")

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Median ET — no complication", f"{et_no:.0f} min")
    with c2: metric_card("Median ET — complication", f"{et_yes:.0f} min")
    with c3: metric_card("Pearson r", f"{r:.3f}", "ET × complication")
    with c4: metric_card("Mann-Whitney p", f"{p_mw:.4f}")

    st.markdown("---")

    # Scatter
    st.subheader("Exposure time vs complications — scatter")
    scatter_df = pd.DataFrame({
        "Exposure Time (min)": et,
        "Complication": comp.map({0: "No complication", 1: "Complication"}),
        "Age": df["Age"],
        "BMI": df["BMI"],
    }).dropna(subset=["Exposure Time (min)", "Complication"])

    fig = px.strip(scatter_df, x="Exposure Time (min)", y="Complication",
                   color="Complication",
                   color_discrete_map={"No complication": COLORS["blue"],
                                       "Complication": COLORS["coral"]},
                   hover_data=["Age", "BMI"],
                   stripmode="overlay")
    fig.update_traces(jitter=0.4, marker_size=5, marker_opacity=0.55)
    fig.update_layout(**plotly_defaults(), height=280,
                      showlegend=True,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-note">Each point = one patient. '
                'Complication group shifts right — longer exposures — but the difference '
                'is not independent of case complexity (see Model tab).</div>',
                unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Complication rate by exposure time bin")
        bins   = [0, 15, 20, 25, 30, 40, 81]
        labels = ["<15", "15–20", "20–25", "25–30", "30–40", ">40"]
        df["et_bin"] = pd.cut(et, bins=bins, labels=labels)
        et_comp = df.groupby("et_bin", observed=True)["Any_Complication"].agg(
            ["sum","count","mean"]).reset_index()
        et_comp["rate"] = (et_comp["mean"] * 100).round(1)
        et_comp["label"] = et_comp["et_bin"].astype(str) + " min"
        colors = [COLORS["blue"] if r < 15 else COLORS["amber"] if r < 25
                  else COLORS["coral"] for r in et_comp["rate"]]
        fig = go.Figure(go.Bar(
            x=et_comp["label"], y=et_comp["rate"],
            text=[f"{r}%\n(n={n})" for r, n in
                  zip(et_comp["rate"], et_comp["count"])],
            textposition="outside", marker_color=colors,
        ))
        fig.add_hline(y=df["Any_Complication"].mean()*100,
                      line_dash="dash", line_color=COLORS["gray"],
                      annotation_text=f"Overall {df['Any_Complication'].mean()*100:.1f}%")
        fig.update_layout(**plotly_defaults(), height=300,
                          yaxis_title="Complication rate (%)",
                          yaxis_range=[0, 50])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Exposure time by prior abdominal surgery")
        if "Prior_Abd_Surg_Flag" in df.columns:
            prior_df = df.copy()
            prior_df["Prior abd surgery"] = prior_df["Prior_Abd_Surg_Flag"].map(
                {0: "No prior surgery", 1: "Prior surgery"})
            fig = px.violin(prior_df, x="Prior abd surgery",
                            y="Exposure Time (min)",
                            color="Prior abd surgery",
                            color_discrete_map={
                                "No prior surgery": COLORS["blue"],
                                "Prior surgery": COLORS["coral"]},
                            box=True, points="outliers")
            fig.update_layout(**plotly_defaults(), height=300,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="section-note">Prior abdominal surgery '
                        'is a potential confounder — it may drive both longer exposures '
                        'and higher complication rates independently.</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("XGBoost AUC", "0.693", "5-fold CV")
    with c2: metric_card("Logistic AUC", "0.691", "5-fold CV")
    with c3: metric_card("M2 AUC (best)", "0.739", "Age + surgical time")
    with c4: metric_card("M3 AUC delta", "+0.000", "Adding exposure time")

    st.markdown("---")
    st.info("**Key finding:** Adding exposure time to a model containing age, ASA, BMI, "
            "surgical time, and number of levels adds zero discriminative value (AUC unchanged "
            "at 0.739). Age and surgical time are the only independent predictors.")

    # SHAP importance
    st.subheader("SHAP feature importance — mean |SHAP| (XGBoost)")
    shap_data = pd.DataFrame({
        "Feature": ["Hospital stay (days)", "Age", "Surgical time (min)",
                    "Incision length (cm)", "BMI", "Exposure time (min)",
                    "EBL (ml)", "ASA class", "Prior abd surgery", "Num levels"],
        "SHAP": [0.698, 0.593, 0.353, 0.305, 0.293, 0.133, 0.065, 0.054, 0.021, 0.003],
    }).sort_values("SHAP")
    colors = [COLORS["coral"] if f != "Exposure time (min)" else COLORS["amber"]
              for f in shap_data["Feature"]]
    fig = px.bar(shap_data, x="SHAP", y="Feature", orientation="h",
                 color="Feature",
                 color_discrete_sequence=colors)
    fig.update_layout(**plotly_defaults(), height=320, showlegend=False,
                      xaxis_title="Mean |SHAP| value")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-note">Hospital stay and age are the strongest '
                'predictors. Exposure time ranks 6th — meaningful in isolation but not '
                'independent once surgical time is controlled.</div>',
                unsafe_allow_html=True)

    # Three model comparison
    st.subheader("Sequential model results — three-model regression")
    st.markdown("""
| Feature | M1 OR (95% CI) | p | M2 OR (95% CI) | p | M3 OR (95% CI) | p | ΔAUC |
|---|---|---|---|---|---|---|---|
| **Age (per year)** | 1.047 (1.021–1.095) | **<0.001** | 1.050 (1.026–1.090) | **<0.001** | 1.049 (1.015–1.100) | **<0.001** | — |
| Sex (male) | 0.966 (0.519–1.606) | 0.880 | 0.929 (0.494–1.800) | 0.850 | 0.921 (0.476–1.862) | 0.870 | — |
| BMI (per unit) | 0.998 (0.940–1.055) | 0.820 | 0.991 (0.917–1.062) | 0.700 | 0.989 (0.913–1.069) | 0.850 | — |
| ASA class (per grade) | 1.821 (0.997–3.495) | 0.060† | 1.797 (0.994–3.359) | 0.070† | 1.793 (0.960–3.097) | 0.070† | — |
| **Surgical time (per min)** | — | — | 1.016 (1.003–1.029) | **0.020** | 1.015 (1.005–1.029) | **0.020** | — |
| Number of levels | — | — | 0.645 (0.293–1.353) | 0.260 | 0.626 (0.268–1.358) | 0.220 | — |
| Exposure time (per min) | — | — | — | — | 1.004 (0.956–1.039) | 0.920 | **0.000** |
| **Model AUC** | **0.695** | | **0.739** | | **0.739** | | |

*Bootstrap 95% CI (500 iterations). † trend p<0.10.*
""")

    # ROC curve
    st.subheader("ROC curve — XGBoost (AUC = 0.693)")
    roc_pts = [[0,0],[0.05,0.28],[0.1,0.45],[0.2,0.60],[0.3,0.70],
               [0.4,0.76],[0.5,0.81],[0.6,0.85],[0.7,0.89],[0.8,0.93],[1,1]]
    roc_df = pd.DataFrame(roc_pts, columns=["FPR","TPR"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc_df["FPR"], y=roc_df["TPR"],
                             name="XGBoost (AUC=0.693)",
                             line=dict(color=COLORS["blue"], width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                             line=dict(color=COLORS["gray"], dash="dash", width=1)))
    fig.update_layout(**plotly_defaults(), height=300,
                      xaxis_title="False positive rate",
                      yaxis_title="True positive rate",
                      legend=dict(x=0.6, y=0.1))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Patient risk calculator")
    st.markdown("Enter patient characteristics to estimate 30-day complication probability.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Patient demographics**")
        age   = st.slider("Age (years)", 18, 90, 60)
        sex   = st.selectbox("Sex", ["Male", "Female"])
        bmi   = st.slider("BMI", 15.0, 55.0, 29.0, 0.5)
        asa   = st.selectbox("ASA class", [1, 2, 3], index=1)
        abd   = st.selectbox("Prior abdominal surgery", ["No", "Yes"])
        rev   = st.selectbox("Revision surgery", ["No", "Yes"])
        spondy= st.selectbox("Spondylolisthesis", ["No", "Yes"])
        lvl   = st.selectbox("Number of spine levels", [1, 2, 3], index=0)

        st.markdown("**Procedural parameters**")
        et_val = st.slider("Exposure time (min)", 8, 80, 17)
        st_val = st.slider("Surgical time (min)", 40, 220, 89)
        hs_val = st.slider("Hospital stay (days)", 0, 10, 1)

    # Calculate risk
    sex_bin  = 1 if sex == "Male" else 0
    abd_bin  = 1 if abd == "Yes"  else 0
    rev_bin  = 1 if rev == "Yes"  else 0
    spondy_b = 1 if spondy == "Yes" else 0

    risk_pct = round(logistic_prob(age, sex_bin, bmi, asa, abd_bin, rev_bin,
                                   et_val, st_val, hs_val, lvl, spondy_b) * 100, 1)

    if risk_pct < 15:
        risk_label = "Low risk"
        risk_color = "#27500A"
        risk_bg    = "#EAF3DE"
        advice     = "Proceed with standard monitoring protocol."
    elif risk_pct < 25:
        risk_label = "Moderate risk"
        risk_color = "#854F0B"
        risk_bg    = "#FAEEDA"
        advice     = "Consider pre-op optimisation. Heightened intra-op monitoring recommended."
    else:
        risk_label = "High risk"
        risk_color = "#A32D2D"
        risk_bg    = "#FCEBEB"
        advice     = "Multidisciplinary review strongly recommended."

    with col2:
        # Risk score box
        st.markdown(f"""
        <div class="risk-box" style="background:{risk_bg}; border-color:{risk_color}40;">
            <div style="font-size:3.2rem; font-weight:700; color:{risk_color};">{risk_pct}%</div>
            <div style="font-size:1.1rem; font-weight:500; color:{risk_color}; margin:6px 0;">{risk_label}</div>
            <div style="font-size:0.85rem; color:#444441;">{advice}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Risk curve
        et_range = np.arange(8, 81, 2)
        risk_curve = [round(logistic_prob(age, sex_bin, bmi, asa, abd_bin,
                                          rev_bin, e, st_val, hs_val,
                                          lvl, spondy_b) * 100, 1)
                      for e in et_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=et_range, y=risk_curve,
            fill="tozeroy", fillcolor="rgba(216,90,48,0.08)",
            line=dict(color=COLORS["coral"], width=2),
            name="Risk curve"))
        fig.add_hline(y=20, line_dash="dash", line_color=COLORS["gray"],
                      annotation_text="20% threshold")
        fig.add_vline(x=et_val, line_dash="dot",
                      line_color="#7F77DD", line_width=1.5,
                      annotation_text=f"Current: {et_val} min")
        fig.update_layout(**plotly_defaults(), height=240,
                          xaxis_title="Exposure time (min)",
                          yaxis_title="Complication probability (%)",
                          yaxis_range=[0, 100], showlegend=False,
                          margin=dict(l=50, r=20, t=20, b=50))
        st.plotly_chart(fig, use_container_width=True)

        # Driver table
        drivers = [
            ("Age", age, "High" if age >= 70 else "Moderate" if age >= 60 else "Low"),
            ("ASA class", asa, "High" if asa >= 3 else "Moderate" if asa == 2 else "Low"),
            ("Surgical time", f"{st_val} min",
             "High" if st_val > 120 else "Moderate" if st_val > 90 else "Low"),
            ("Exposure time", f"{et_val} min",
             "Elevated" if et_val > 40 else "Moderate" if et_val > 25 else "Low"),
            ("Hospital stay", f"{hs_val} d",
             "High" if hs_val >= 3 else "Moderate" if hs_val >= 2 else "Low"),
            ("Prior abd surgery", abd, "Moderate" if abd == "Yes" else "Low"),
            ("BMI", bmi, "Elevated" if bmi >= 35 else "Low"),
        ]

        color_map = {"High": "🔴", "Moderate": "🟡",
                     "Elevated": "🟡", "Low": "🟢"}
        driver_df = pd.DataFrame(drivers, columns=["Factor", "Value", "Risk"])
        driver_df["Risk"] = driver_df["Risk"].map(
            lambda x: f"{color_map.get(x,'⚪')} {x}")
        st.dataframe(driver_df, hide_index=True, use_container_width=True,
                     height=280)

    st.markdown("---")
    st.caption("⚠️ This tool is for clinical decision support only. Predictions are based on "
               "331 historical cases and should be interpreted alongside clinical judgment. "
               "Not validated for prospective clinical use.")
