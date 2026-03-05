# 🌿 Green Luxury Signal Intelligence Suite

An interactive Streamlit dashboard performing **Descriptive, Diagnostic, Predictive, and Prescriptive**
analysis on Gen Z sustainable luxury purchasing behaviour — answering the central question:
**"Which brand signals make Gen Z choose sustainable luxury?"**

---

## 📊 Analysis Framework

| Tab | Question | Techniques |
|---|---|---|
| **Descriptive** | What does the data show? | KPI cards, distributions, sunburst drill-downs, signal radar |
| **Diagnostic** | Why does it happen? | Correlation analysis, Chi-Square + Cramér's V, Signal Gap t-tests, risk factor combos |
| **Predictive** | What will happen? | Logistic Regression, Random Forest, Gradient Boosting, ROC curves, feature importance |
| **Prescriptive** | What should brands do? | Propensity simulator, strategic recommendations, Impact–Cost matrix |

---

## 📁 Project Structure

```
green-luxury-dashboard/
├── app.py                   # Main Streamlit application
├── Final-Sheet1-1.csv       # Survey dataset (123 respondents, 17 variables)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit theme (dark, green accent)
└── README.md
```

---

## 🚀 Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/green-luxury-dashboard.git
cd green-luxury-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy!

---

## 📦 Dataset

- **123 respondents** · **17 variables**
- Target variable: `Sustainable Choice Freq` (Never → Always)
- Brand signal constructs: Environmental Certifications, Sustainability Storytelling, Premium Pricing,
  Eco Materials, Sourcing Transparency, Heritage Credibility
- Greenwashing Skepticism: 2-item scale
- Demographics: Age, Education, Employment Status, Industry

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Streamlit** | Dashboard framework |
| **Plotly** | Interactive visualisations |
| **scikit-learn** | Predictive models |
| **SciPy** | Chi-Square & t-tests |
| **Pandas / NumPy** | Data processing |

---

## 📖 Research Foundation

Based on: *"The Psychology of Green Luxury: How Brand Signals Shape Gen Z's Green Luxury Purchase Decisions"*
— Isha Srivastava, GMBA GS25NS042, Applied Research Project Term I, February 2026

Key findings incorporated:
- Sustainability Storytelling: strongest positive predictor (β=0.119, p=0.004)
- Environmental Certifications: significant positive predictor (β=0.076, p=0.043)
- Premium Price Justification: significant negative predictor (β=−0.102, p=0.015)
- Model R² = 12.0% (consistent with exploratory consumer behaviour research)

---

MIT License · Free to use, modify, and distribute
