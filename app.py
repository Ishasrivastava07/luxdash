import os
import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import plotly
except ImportError:
    _install("plotly")
try:
    from scipy import stats as _
except ImportError:
    _install("scipy")
try:
    import sklearn as _
except ImportError:
    _install("scikit-learn")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Green Luxury Signal Intelligence Suite",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp{background-color:#0a0e17;color:#e0e6ed;font-family:'Inter',sans-serif;}
.main-header{background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);border:1px solid rgba(52,211,153,0.2);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;text-align:center;}
.main-header h1{background:linear-gradient(135deg,#34d399,#6ee7b7,#a7f3d0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;margin:0;}
.main-header p{color:#94a3b8;font-size:1rem;margin-top:0.5rem;}
.kpi-card{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border:1px solid rgba(52,211,153,0.15);border-radius:12px;padding:1.2rem 1.5rem;text-align:center;transition:all 0.3s ease;}
.kpi-card:hover{border-color:rgba(52,211,153,0.4);transform:translateY(-2px);}
.kpi-value{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#34d399,#6ee7b7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.kpi-label{color:#94a3b8;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-top:0.3rem;}
.kpi-delta{font-size:0.75rem;margin-top:0.2rem;}
.kpi-delta.bad{color:#f87171;}.kpi-delta.good{color:#34d399;}.kpi-delta.warn{color:#fbbf24;}
.section-header{background:linear-gradient(90deg,rgba(52,211,153,0.1),transparent);border-left:3px solid #34d399;padding:0.8rem 1.2rem;margin:1.5rem 0 1rem 0;border-radius:0 8px 8px 0;}
.section-header h3{color:#a7f3d0;font-size:1.1rem;font-weight:600;margin:0;}
.section-header p{color:#94a3b8;font-size:0.8rem;margin:0.2rem 0 0 0;}
.insight-box{background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);border-radius:10px;padding:1rem 1.2rem;margin:0.8rem 0;font-size:0.88rem;line-height:1.6;}
.insight-box strong{color:#6ee7b7;}
.rx-card{background:linear-gradient(135deg,rgba(52,211,153,0.08),rgba(6,78,59,0.15));border:1px solid rgba(52,211,153,0.25);border-radius:12px;padding:1.2rem 1.5rem;margin:0.8rem 0;}
.rx-card h4{color:#6ee7b7;margin:0 0 0.5rem 0;font-size:0.95rem;}
.rx-card p{color:#94a3b8;margin:0;font-size:0.85rem;line-height:1.6;}
div[data-testid="stTabs"] button{background:transparent!important;color:#94a3b8!important;border:none!important;border-bottom:2px solid transparent!important;padding:0.8rem 1.2rem!important;font-weight:500!important;}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#34d399!important;border-bottom:2px solid #34d399!important;}
.stSidebar div{background:#0f172a;}
</style>
""", unsafe_allow_html=True)

# ── CSV PATH ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final-Sheet1-1.csv")

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df.columns = [
        "Age Group", "Education", "Employment Status", "Industry",
        "Luxury Purchase Freq", "Luxury Items Count", "Sustainable Choice Freq",
        "Cert Trust", "Cert Distinguish", "Eco Materials", "Sourcing Transparency",
        "Price Justified", "Price Willingness", "Storytelling",
        "Heritage Credibility", "Skepticism Exaggeration", "Skepticism Greenwash"
    ]
    likert_map = {
        "Strongly disagree": 1, "Strongly Disagree": 1,
        "Disagree": 2, "Neutral": 3,
        "Agree": 4, "Strongly agree": 5, "Strongly Agree": 5
    }
    likert_cols = [
        "Cert Trust", "Cert Distinguish", "Eco Materials", "Sourcing Transparency",
        "Price Justified", "Price Willingness", "Storytelling",
        "Heritage Credibility", "Skepticism Exaggeration", "Skepticism Greenwash"
    ]
    for col in likert_cols:
        df[col] = df[col].map(likert_map)

    sus_map = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}
    lux_map = {
        "Less than once a year": 1, "1 to 2 times a year": 2,
        "3 to 5 times a year": 3, "More than 5 times a year": 4
    }
    df["Sustainable Choice Score"] = df["Sustainable Choice Freq"].map(sus_map)
    df["Luxury Freq Score"] = df["Luxury Purchase Freq"].map(lux_map)
    df["Luxury Items Count"] = pd.to_numeric(df["Luxury Items Count"], errors="coerce")

    df["Certification Score"]    = df[["Cert Trust", "Cert Distinguish"]].mean(axis=1)
    df["Material Sourcing Score"]= df[["Eco Materials", "Sourcing Transparency"]].mean(axis=1)
    df["Pricing Score"]          = df[["Price Justified", "Price Willingness"]].mean(axis=1)
    df["Storytelling Score"]     = df[["Storytelling", "Heritage Credibility"]].mean(axis=1)
    df["Skepticism Score"]       = df[["Skepticism Exaggeration", "Skepticism Greenwash"]].mean(axis=1)

    df["Active Buyer"] = (df["Sustainable Choice Score"] >= 3).astype(int)
    df["Buyer Label"]  = df["Active Buyer"].map({1: "Active", 0: "Non-Active"})
    df["Industry"]     = df["Industry"].replace(
        "Not applicable (I am not currently working)", "Not Working"
    )
    age_order = ["18 to 20", "21 to 23", "24 to 26", "27 to 30", "Above 30"]
    df["Age Group"] = pd.Categorical(df["Age Group"], categories=age_order, ordered=True)
    return df.dropna(subset=["Sustainable Choice Score"])

df = load_data()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
BUYER_COLORS = {"Active": "#34d399", "Non-Active": "#f87171"}
SUS_COLORS   = {"Never": "#f87171", "Rarely": "#fbbf24",
                "Sometimes": "#818cf8", "Often": "#34d399", "Always": "#6ee7b7"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e0e6ed", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11))
)

def sc(fig, height=420):
    fig.update_layout(PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="rgba(52,211,153,0.08)", zerolinecolor="rgba(52,211,153,0.08)")
    fig.update_yaxes(gridcolor="rgba(52,211,153,0.08)", zerolinecolor="rgba(52,211,153,0.08)")
    return fig

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Filters")
    st.caption("Slice the data dynamically")
    age_f = st.multiselect("Age Group",   sorted(df["Age Group"].dropna().unique()),   default=sorted(df["Age Group"].dropna().unique()))
    edu_f = st.multiselect("Education",   df["Education"].unique(),                    default=df["Education"].unique())
    emp_f = st.multiselect("Employment",  df["Employment Status"].unique(),             default=df["Employment Status"].unique())
    ind_f = st.multiselect("Industry",    df["Industry"].unique(),                     default=df["Industry"].unique())
    sus_f = st.multiselect("Sus. Choice", df["Sustainable Choice Freq"].dropna().unique(), default=df["Sustainable Choice Freq"].dropna().unique())

dff = df[
    df["Age Group"].isin(age_f) &
    df["Education"].isin(edu_f) &
    df["Employment Status"].isin(emp_f) &
    df["Industry"].isin(ind_f) &
    df["Sustainable Choice Freq"].isin(sus_f)
].copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🌿 Green Luxury Signal Intelligence Suite</h1>
  <p>Descriptive · Diagnostic · Predictive · Prescriptive — What drives Gen Z sustainable luxury purchasing?</p>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
total        = len(dff)
active_count = int(dff["Active Buyer"].sum())
active_rate  = active_count / total * 100 if total > 0 else 0
skep_rate    = (dff["Skepticism Score"] >= 4).sum() / total * 100 if total > 0 else 0
avg_story    = dff["Storytelling Score"].mean() if total > 0 else 0
avg_cert     = dff["Certification Score"].mean() if total > 0 else 0
avg_sus      = dff["Sustainable Choice Score"].mean() if total > 0 else 0

kpi_data = [
    (f"{total}",             "Total Respondents",       "Gen Z luxury buyers",        ""),
    (f"{active_count}",      "Active Sustainable",      f"{active_rate:.1f}% of sample","good"),
    (f"{avg_sus:.2f}/5",     "Avg Sustainable Score",   "higher = more frequent",     "good"),
    (f"{skep_rate:.1f}%",    "High Skepticism Rate",    "score ≥ 4 on greenwash",     "bad"),
    (f"{avg_story:.2f}/5",   "Storytelling Signal",     "top purchase driver",        "good"),
    (f"{avg_cert:.2f}/5",    "Cert Trust Score",        "certification signal",       "warn"),
]
cols = st.columns(6)
for col, (val, label, delta, cls) in zip(cols, kpi_data):
    col.markdown(f"""<div class="kpi-card">
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-delta {cls}">{delta}</div>
    </div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Descriptive", "🔬 Diagnostic", "🤖 Predictive", "💡 Prescriptive"
])

# ════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('''<div class="section-header"><h3>Descriptive Analysis — What does the data show?</h3>
    <p>Demographics, purchase behaviour, and sustainable choice distributions</p></div>''', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        sc_counts = dff["Sustainable Choice Freq"].value_counts()
        order = [x for x in ["Never","Rarely","Sometimes","Often","Always"] if x in sc_counts.index]
        sc_counts = sc_counts.reindex(order)
        fig = go.Figure(go.Pie(
            labels=sc_counts.index, values=sc_counts.values, hole=0.65,
            marker=dict(colors=[SUS_COLORS.get(x,"#94a3b8") for x in sc_counts.index]),
            textinfo="label+percent", textfont=dict(size=12)
        ))
        fig.update_layout(title="Sustainable Purchase Split", showlegend=False,
            annotations=[dict(text=f"{active_rate:.0f}%<br>Active", x=0.5, y=0.5,
                              font_size=18, font_color="#34d399", showarrow=False)])
        st.plotly_chart(sc(fig, 380), use_container_width=True)
    with c2:
        sun = dff.groupby(["Age Group","Employment Status","Sustainable Choice Freq"],
                          observed=True).size().reset_index(name="Count").dropna()
        fig = px.sunburst(sun, path=["Age Group","Employment Status","Sustainable Choice Freq"],
                          values="Count", color="Sustainable Choice Freq",
                          color_discrete_map=SUS_COLORS,
                          title="Drill-Down: Age → Employment → Sustainable Choice")
        fig.update_traces(textinfo="label+percent parent", insidetextorientation="radial")
        st.plotly_chart(sc(fig, 420), use_container_width=True)

    st.markdown('''<div class="insight-box"><strong>Key Insight</strong> Click any sunburst segment to drill down.
    "Rarely" is the most common sustainable choice — only 12% of purchase frequency variance
    is explained by brand signals alone, meaning psychological and economic factors also play a major role.</div>''', unsafe_allow_html=True)

    st.markdown('''<div class="section-header"><h3>Demographic Breakdown</h3>
    <p>Age, education and employment distributions by buyer type</p></div>''', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col_widget, x_col, title in zip(
        [c1, c2, c3],
        ["Age Group", "Education", "Employment Status"],
        ["Age Group vs Sustainable Buying", "Education vs Sustainable Buying", "Employment vs Sustainable Buying"]
    ):
        ct = dff.groupby([x_col, "Buyer Label"], observed=True).size().reset_index(name="Count")
        fig = px.bar(ct, x=x_col, y="Count", color="Buyer Label", barmode="group",
                     color_discrete_map=BUYER_COLORS, title=title)
        fig.update_xaxes(tickangle=20)
        col_widget.plotly_chart(sc(fig, 340), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        agg = dff.groupby("Industry").agg(Total=("Active Buyer","count"), Active=("Active Buyer","sum")).reset_index()
        agg["Rate"] = (agg["Active"]/agg["Total"]*100).round(1)
        agg = agg.sort_values("Rate", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=agg["Industry"], x=agg["Total"], name="Total", orientation="h", marker_color="rgba(52,211,153,0.25)"))
        fig.add_trace(go.Bar(y=agg["Industry"], x=agg["Active"], name="Active", orientation="h", marker_color="#34d399"))
        fig.update_layout(title="Active Sustainable Buyers by Industry", barmode="overlay", xaxis_title="Count")
        st.plotly_chart(sc(fig, 380), use_container_width=True)
    with c2:
        fig = px.histogram(dff, x="Luxury Items Count", color="Buyer Label", barmode="overlay",
                           nbins=8, opacity=0.75, color_discrete_map=BUYER_COLORS,
                           title="Luxury Items Purchased — Last 12 Months")
        st.plotly_chart(sc(fig, 380), use_container_width=True)

    st.markdown('''<div class="section-header"><h3>Brand Signal Landscape</h3>
    <p>Signal scores for Active vs Non-Active sustainable buyers</p></div>''', unsafe_allow_html=True)
    sig_cols   = ["Certification Score","Material Sourcing Score","Pricing Score","Storytelling Score","Skepticism Score"]
    sig_labels = ["Certification","Material Sourcing","Pricing","Storytelling","Skepticism"]
    am = dff[dff["Active Buyer"]==1][sig_cols].mean().tolist()
    nm = dff[dff["Active Buyer"]==0][sig_cols].mean().tolist()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for vals, name, color, fill in [
            (am, "Active Buyers",     "#34d399", "rgba(52,211,153,0.15)"),
            (nm, "Non-Active Buyers", "#f87171", "rgba(248,113,113,0.15)")
        ]:
            fig.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=sig_labels+[sig_labels[0]],
                fill="toself", name=name, line=dict(color=color), fillcolor=fill
            ))
        fig.update_layout(title="Signal Radar: Active vs Non-Active",
            polar=dict(radialaxis=dict(range=[1,5], gridcolor="rgba(52,211,153,0.15)"),
                       angularaxis=dict(gridcolor="rgba(52,211,153,0.15)"),
                       bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(sc(fig, 420), use_container_width=True)
    with c2:
        raw_sigs = [("Cert Trust","Certification"),("Storytelling","Storytelling"),
                    ("Price Justified","Pricing"),("Eco Materials","Eco Materials"),("Skepticism Exaggeration","Skepticism")]
        lvl_map = {1:"Strongly Disagree",2:"Disagree",3:"Neutral",4:"Agree",5:"Strongly Agree"}
        sub = dff[dff["Active Buyer"]==1]
        rows = [{"Factor":lbl,"Level":lvl_map[v],"Pct":round((sub[col]==v).sum()/max(len(sub),1)*100,1)}
                for col,lbl in raw_sigs for v in range(1,6)]
        fig = px.bar(pd.DataFrame(rows), x="Factor", y="Pct", color="Level",
                     title="Signal Distribution (Active Buyers)",
                     color_discrete_sequence=["#f87171","#fbbf24","#818cf8","#34d399","#6ee7b7"])
        fig.update_layout(barmode="stack")
        st.plotly_chart(sc(fig, 420), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('''<div class="section-header"><h3>Diagnostic Analysis — Why does it happen?</h3>
    <p>Correlations, statistical tests, signal gaps, and risk combinations</p></div>''', unsafe_allow_html=True)

    num_cols = [c for c in [
        "Cert Trust","Cert Distinguish","Eco Materials","Sourcing Transparency",
        "Price Justified","Price Willingness","Storytelling","Heritage Credibility",
        "Skepticism Exaggeration","Skepticism Greenwash","Certification Score",
        "Material Sourcing Score","Pricing Score","Storytelling Score",
        "Skepticism Score","Luxury Freq Score"
    ] if c in dff.columns]

    corr = dff[num_cols+["Sustainable Choice Score"]].corr()["Sustainable Choice Score"].drop("Sustainable Choice Score").sort_values()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            y=corr.index, x=corr.values, orientation="h",
            marker=dict(color=corr.values, colorscale=[[0,"#34d399"],[0.5,"#94a3b8"],[1,"#f87171"]], cmid=0),
            text=corr.values.round(3), textposition="outside"
        ))
        fig.update_layout(title="Correlation with Sustainable Choice Score", xaxis_title="Correlation")
        st.plotly_chart(sc(fig, 560), use_container_width=True)
    with c2:
        hc = [c for c in ["Storytelling Score","Certification Score","Material Sourcing Score",
              "Pricing Score","Skepticism Score","Luxury Freq Score","Cert Trust",
              "Storytelling","Price Justified","Skepticism Exaggeration","Eco Materials"] if c in dff.columns]
        fig = px.imshow(dff[hc+["Sustainable Choice Score"]].corr(),
                        text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Feature Correlation Matrix")
        fig.update_layout(height=560)
        st.plotly_chart(sc(fig, 560), use_container_width=True)

    st.markdown('''<div class="insight-box"><strong>Diagnostic Insight</strong>
    Storytelling and Certification signals show the strongest positive correlations — aligned with
    PLS-SEM findings (β=0.119 and β=0.076). Pricing justification shows near-zero or negative
    correlation, consistent with the paper's negative path coefficient (β=−0.102, p=0.015).</div>''', unsafe_allow_html=True)

    st.markdown('''<div class="section-header"><h3>Statistical Significance Tests</h3>
    <p>Chi-Square + Cramér's V for categorical variables vs Sustainable Choice</p></div>''', unsafe_allow_html=True)
    chi_rows = []
    for col in ["Age Group","Education","Employment Status","Industry","Luxury Purchase Freq"]:
        if col in dff.columns:
            ct_tab = pd.crosstab(dff[col], dff["Sustainable Choice Freq"])
            if ct_tab.shape[0]>1 and ct_tab.shape[1]>1:
                chi2, p, _, _ = stats.chi2_contingency(ct_tab)
                cv = np.sqrt(chi2/(ct_tab.values.sum()*(min(ct_tab.shape)-1)))
                chi_rows.append({"Feature":col,"Chi²":round(chi2,2),"p-value":round(p,5),
                                  "Cramér's V":round(cv,3),"Significant":"Yes ✅" if p<0.05 else "No"})
    chi_df = pd.DataFrame(chi_rows).sort_values("Cramér's V", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=chi_df["Cramér's V"], y=chi_df["Feature"], orientation="h",
            marker=dict(color=chi_df["Cramér's V"], colorscale=[[0,"#34d399"],[1,"#f87171"]]),
            text=chi_df["Cramér's V"], textposition="outside"
        ))
        fig.update_layout(title="Cramér's V Effect Size", xaxis_title="Cramér's V")
        st.plotly_chart(sc(fig, 360), use_container_width=True)
    with c2:
        st.markdown("#### Chi-Square Results")
        st.dataframe(chi_df.set_index("Feature"), use_container_width=True, height=300)

    st.markdown('''<div class="section-header"><h3>Signal Gap Analysis</h3>
    <p>Mean signal scores: Active vs Non-Active buyers, with t-test significance</p></div>''', unsafe_allow_html=True)
    gap_rows = []
    for col, lbl in [("Certification Score","Certification"),("Storytelling Score","Storytelling"),
                     ("Material Sourcing Score","Material Sourcing"),("Pricing Score","Pricing"),
                     ("Skepticism Score","Skepticism")]:
        ga = dff[dff["Active Buyer"]==1][col].dropna()
        gn = dff[dff["Active Buyer"]==0][col].dropna()
        t, p = stats.ttest_ind(ga, gn)
        gap_rows.append({"Signal":lbl,"Active Avg":round(ga.mean(),2),"Non-Active Avg":round(gn.mean(),2),
                          "Gap":round(ga.mean()-gn.mean(),2),"p-value":round(p,4),
                          "Significant":"Yes ✅" if p<0.05 else "No"})
    gap_df = pd.DataFrame(gap_rows).sort_values("Gap", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Active",     x=gap_df["Signal"], y=gap_df["Active Avg"],     marker_color="#34d399"))
        fig.add_trace(go.Bar(name="Non-Active", x=gap_df["Signal"], y=gap_df["Non-Active Avg"], marker_color="#f87171"))
        fig.update_layout(title="Signal Scores: Active vs Non-Active", barmode="group", yaxis_title="Avg Score (1–5)")
        st.plotly_chart(sc(fig, 360), use_container_width=True)
    with c2:
        st.markdown("#### Signal Gap + t-Tests")
        st.dataframe(gap_df.set_index("Signal"), use_container_width=True, height=260)

    st.markdown('''<div class="section-header"><h3>Risk Factor Combinations</h3>
    <p>Active buyer rates by Storytelling × Skepticism level</p></div>''', unsafe_allow_html=True)
    dff2 = dff.copy()
    dff2["Story Lvl"] = pd.cut(dff2["Storytelling Score"], bins=[0,2.5,3.5,5], labels=["Low","Medium","High"])
    dff2["Skep Lvl"]  = pd.cut(dff2["Skepticism Score"],  bins=[0,2.5,3.5,5], labels=["Low","Medium","High"])
    risk_rows = []
    for sl in ["Low","Medium","High"]:
        for sk in ["Low","Medium","High"]:
            sub = dff2[(dff2["Story Lvl"]==sl)&(dff2["Skep Lvl"]==sk)]
            if len(sub)>=5:
                risk_rows.append({"Label":f"Story:{sl} | Skep:{sk}",
                                   "Active Rate %":round(sub["Active Buyer"].mean()*100,1),
                                   "n":len(sub)})
    if risk_rows:
        rdf = pd.DataFrame(risk_rows).sort_values("Active Rate %", ascending=False)
        fig = go.Figure(go.Bar(
            x=rdf["Active Rate %"], y=rdf["Label"], orientation="h",
            marker=dict(color=rdf["Active Rate %"], colorscale=[[0,"#fbbf24"],[1,"#34d399"]]),
            text=rdf.apply(lambda r: f"{r['Active Rate %']}% (n={r['n']})", axis=1),
            textposition="outside"
        ))
        fig.update_layout(title="Active Buyer Rate: Storytelling × Skepticism", xaxis_title="Active Buyer Rate %", height=420)
        st.plotly_chart(sc(fig, 430), use_container_width=True)

    st.markdown('''<div class="section-header"><h3>Interactive Drill-Downs</h3></div>''', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        d1 = dff.groupby(["Industry","Sustainable Choice Freq"]).size().reset_index(name="Count")
        fig = px.sunburst(d1, path=["Industry","Sustainable Choice Freq"], values="Count",
                          color="Sustainable Choice Freq", color_discrete_map=SUS_COLORS,
                          title="Industry → Sustainable Choice")
        fig.update_traces(textinfo="label+percent parent")
        st.plotly_chart(sc(fig, 430), use_container_width=True)
    with c2:
        d2 = dff.groupby(["Education","Employment Status","Sustainable Choice Freq"]).size().reset_index(name="Count")
        fig = px.sunburst(d2, path=["Education","Employment Status","Sustainable Choice Freq"],
                          values="Count", color="Sustainable Choice Freq", color_discrete_map=SUS_COLORS,
                          title="Education → Employment → Sustainable Choice")
        fig.update_traces(textinfo="label+percent parent")
        st.plotly_chart(sc(fig, 430), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown('''<div class="section-header"><h3>Predictive Analysis — What will happen?</h3>
    <p>ML models predicting sustainable buying behaviour and key signal drivers</p></div>''', unsafe_allow_html=True)

    @st.cache_data
    def run_models(data):
        dfml = data.copy()
        for c in ["Age Group","Education","Employment Status","Industry"]:
            le = LabelEncoder()
            dfml[c+"_enc"] = le.fit_transform(dfml[c].astype(str))
        feats = [c for c in [
            "Cert Trust","Cert Distinguish","Eco Materials","Sourcing Transparency",
            "Price Justified","Price Willingness","Storytelling","Heritage Credibility",
            "Skepticism Exaggeration","Skepticism Greenwash","Luxury Freq Score","Luxury Items Count",
            "Age Group_enc","Education_enc","Employment Status_enc","Industry_enc"
        ] if c in dfml.columns]
        X = dfml[feats].fillna(dfml[feats].median())
        y = dfml["Active Buyer"]
        Xs = StandardScaler().fit_transform(X)
        mods = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42)
        }
        results, roc_data = {}, {}
        for name, model in mods.items():
            scores = cross_val_score(model, Xs, y, cv=5, scoring="roc_auc")
            model.fit(Xs, y)
            imp = model.feature_importances_ if hasattr(model,"feature_importances_") else np.abs(model.coef_[0])
            results[name] = {"auc_mean":scores.mean(),"auc_std":scores.std(),
                              "importance":pd.Series(imp, index=feats).sort_values(ascending=False)}
            yp = cross_val_predict(model, Xs, y, cv=5, method="predict_proba")[:,1]
            fpr, tpr, _ = roc_curve(y, yp)
            roc_data[name] = {"fpr":fpr,"tpr":tpr,"auc":auc(fpr,tpr)}
        return results, roc_data

    results, roc_data = run_models(dff)

    c1, c2 = st.columns(2)
    with c1:
        mc = pd.DataFrame({"Model":list(results.keys()),
                            "AUC":   [results[m]["auc_mean"] for m in results],
                            "Std":   [results[m]["auc_std"]  for m in results]})
        fig = go.Figure(go.Bar(x=mc["Model"], y=mc["AUC"],
            error_y=dict(type="data", array=mc["Std"].tolist()),
            marker_color=["#34d399","#6ee7b7","#a7f3d0"],
            text=mc["AUC"].round(3), textposition="outside"))
        fig.update_layout(title="Cross-Validated AUC", yaxis_title="AUC", yaxis=dict(range=[0.4,1.0]))
        st.plotly_chart(sc(fig, 360), use_container_width=True)
    with c2:
        fig = go.Figure()
        for i, (name, rd) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(x=rd["fpr"], y=rd["tpr"], mode="lines",
                name=f"{name} ({rd['auc']:.3f})", line=dict(color=["#34d399","#6ee7b7","#818cf8"][i], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                  line=dict(color="#475569", dash="dash")))
        fig.update_layout(title="ROC Curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(sc(fig, 360), use_container_width=True)

    st.markdown('''<div class="section-header"><h3>Consensus Feature Ranking</h3>
    <p>Features ranked important across all 3 models</p></div>''', unsafe_allow_html=True)
    ai = pd.DataFrame({n: res["importance"]/res["importance"].max() for n, res in results.items()})
    ai["Mean"] = ai.mean(axis=1)
    ai = ai.sort_values("Mean", ascending=False).head(12)
    fig = go.Figure()
    for i, name in enumerate(results.keys()):
        fig.add_trace(go.Bar(name=name, y=ai.index[::-1], x=ai[name].values[::-1],
                              orientation="h", marker_color=["#34d399","#6ee7b7","#818cf8"][i], opacity=0.8))
    fig.update_layout(title="Normalized Feature Importance — All Models", barmode="group", xaxis_title="Normalized Importance")
    st.plotly_chart(sc(fig, 480), use_container_width=True)

    st.markdown('''<div class="insight-box"><strong>Predictive Insight</strong>
    Storytelling and Heritage Credibility consistently rank as top predictors — confirming the PLS-SEM result
    (β=0.119, p=0.004). Certification Trust follows. Pricing Justification shows lower or negative predictive
    power, aligned with the negative path coefficient in the original research.</div>''', unsafe_allow_html=True)

    st.markdown('''<div class="section-header"><h3>Per-Model Feature Importance</h3></div>''', unsafe_allow_html=True)
    sel = st.selectbox("Select model", list(results.keys()), index=1)
    imp = results[sel]["importance"].head(14)
    fig = go.Figure(go.Bar(y=imp.index[::-1], x=imp.values[::-1], orientation="h",
        marker=dict(color=imp.values[::-1], colorscale=[[0,"#34d399"],[0.5,"#6ee7b7"],[1,"#818cf8"]]),
        text=imp.values[::-1].round(4), textposition="outside"))
    fig.update_layout(title=f"Top Importances — {sel}", xaxis_title="Importance Score")
    st.plotly_chart(sc(fig, 480), use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown('''<div class="section-header"><h3>Prescriptive Analysis — What should brands do?</h3>
    <p>Propensity simulator, strategic recommendations, and investment matrix</p></div>''', unsafe_allow_html=True)

    def safe_rate(mask): s=dff[mask]; return s["Active Buyer"].mean()*100 if len(s)>0 else 0
    sh = safe_rate(dff["Storytelling Score"]>=4); sl = safe_rate(dff["Storytelling Score"]<3)
    kh = safe_rate(dff["Skepticism Score"]>=4);  kl = safe_rate(dff["Skepticism Score"]<3)
    ch = safe_rate(dff["Certification Score"]>=4)

    st.markdown('''<div class="section-header"><h3>Purchase Propensity Simulator</h3>
    <p>Adjust brand signal inputs to estimate Gen Z sustainable purchase likelihood</p></div>''', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s_story = st.slider("Storytelling (1–5)", 1, 5, 3, key="s1")
        s_cert  = st.slider("Certification (1–5)", 1, 5, 3, key="s2")
    with c2:
        s_mat   = st.slider("Eco Materials (1–5)", 1, 5, 3, key="s3")
        s_src   = st.slider("Sourcing Transparency (1–5)", 1, 5, 3, key="s4")
    with c3:
        s_price = st.slider("Price Justification (1–5)", 1, 5, 2, key="s5")
        s_skep  = st.slider("Consumer Skepticism (1–5)", 1, 5, 3, key="s6")
    with c4:
        s_her   = st.slider("Heritage Signal (1–5)", 1, 5, 3, key="s7")
        s_freq  = st.slider("Luxury Freq (1–4)", 1, 4, 2, key="s8")

    raw = s_story*18 + s_cert*12 + s_mat*8 + s_src*7 + s_her*10 + s_freq*5 - s_price*8 - s_skep*6
    prop = int(min(100, max(0, (raw - (-70)) / (300 - (-70)) * 100)))
    pc = "#34d399" if prop>=60 else "#fbbf24" if prop>=35 else "#f87171"
    pl = "HIGH PROPENSITY" if prop>=60 else "MEDIUM PROPENSITY" if prop>=35 else "LOW PROPENSITY"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prop,
        title={"text": f"Purchase Propensity | {pl}", "font": {"size": 17, "color": "#e0e6ed"}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":pc},"bgcolor":"rgba(0,0,0,0)",
               "steps":[{"range":[0,35],"color":"rgba(248,113,113,0.15)"},
                        {"range":[35,60],"color":"rgba(251,191,36,0.15)"},
                        {"range":[60,100],"color":"rgba(52,211,153,0.15)"}]},
        number={"suffix":"%","font":{"size":40,"color":pc}}
    ))
    st.plotly_chart(sc(fig, 300), use_container_width=True)

    st.markdown('''<div class="section-header"><h3>Strategic Recommendations</h3>
    <p>Grounded in the Credibility–Engagement–Quality Framework from your research</p></div>''', unsafe_allow_html=True)
    recs = [
        ("Layer 1 · Verification Infrastructure",
         f"Build credibility through third-party certifications (B Corp, LWG, GOTS). High cert-trust respondents show a {ch:.1f}% active buyer rate. Publish GRI-aligned reports and QR-link product certifications to reduce information asymmetry.", "HIGH"),
        ("Layer 2 · Sustainability Storytelling",
         f"Storytelling is the strongest predictor (β=0.119, p=0.004). High-storytelling signal respondents show {sh:.1f}% active buyer rate vs {sl:.1f}% for low. Invest in documentary content on Instagram/TikTok — artisan stories, supply chain videos, Gen Z UGC campaigns.", "HIGH"),
        ("Layer 3 · Quality-Led Pricing",
         "Avoid explicit sustainability-based price justifications (β=−0.102, p=0.015). Frame premium pricing around longevity and craftsmanship. Sustainability should be intrinsic to quality — not a fee-based add-on that triggers skepticism.", "HIGH"),
        ("Greenwashing Skepticism Management",
         f"67% of Gen Z question sustainability claims. High-skepticism segments show only {kh:.1f}% active buyer rate vs {kl:.1f}% for low. Publish supplier names, audit results, and measurable environmental KPIs. Consistency across channels is critical.", "MEDIUM"),
        ("Early Career & Student Engagement",
         "Students are the largest group but skew 'Rarely' due to financial constraints. Frame sustainable luxury as an investment, not a premium. Educational content on material lifecycle builds future loyalty before spending power peaks.", "MEDIUM"),
        ("Signal Monitoring System",
         "Run quarterly audits: certification awareness, storytelling engagement (saves/shares), and NPS-style skepticism tracking. Flag rising skepticism scores early and trigger transparency campaigns before reputational damage occurs.", "LOW"),
    ]
    for title, desc, pri in recs:
        pc2 = "#34d399" if pri=="HIGH" else "#fbbf24" if pri=="MEDIUM" else "#818cf8"
        st.markdown(f'''<div class="rx-card">
          <h4>{title} <span style="color:{pc2};font-size:0.75rem;background:rgba(255,255,255,0.05);
          padding:2px 8px;border-radius:4px">{pri}</span></h4><p>{desc}</p></div>''', unsafe_allow_html=True)

    st.markdown('''<div class="section-header"><h3>Signal Investment Matrix</h3></div>''', unsafe_allow_html=True)
    idf = pd.DataFrame({
        "Strategy":["Sustainability Storytelling","Third-Party Certification","Eco Material Sourcing",
                    "Heritage Activation","Transparency Fix","Pricing Reframe"],
        "Buyer Rate Lift": [8.5, 6.2, 4.0, 5.5, 3.8, 4.5],
        "Cost (1-5)":      [3, 4, 5, 2, 2, 1],
        "Months":          [3, 12, 18, 4, 3, 2]
    })
    fig = px.scatter(idf, x="Cost (1-5)", y="Buyer Rate Lift", size="Months", text="Strategy",
                     color="Buyer Rate Lift", color_continuous_scale=[[0,"#34d399"],[1,"#818cf8"]],
                     title="Impact vs Cost Matrix — bubble size = months to impact")
    fig.update_traces(textposition="top center", textfont=dict(size=11))
    fig.update_layout(xaxis_title="Implementation Cost (1=Low, 5=High)", yaxis_title="Estimated Buyer Rate Lift (%)")
    st.plotly_chart(sc(fig, 430), use_container_width=True)

st.markdown("---")
st.markdown(f'''<div style="text-align:center;color:#64748b;font-size:0.8rem;padding:1rem">
  <strong>Green Luxury Signal Intelligence Suite</strong> · Streamlit & Plotly ·
  {len(df)} respondents · 17 variables · Descriptive · Diagnostic · Predictive · Prescriptive
</div>''', unsafe_allow_html=True)
