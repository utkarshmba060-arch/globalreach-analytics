# 🌍 GlobalReach Analytics Dashboard

**Data Analytics Individual Assignment**  
**Student:** Utkarsh Chandra  
**Institution:** SP Jain School of Global Management  
**Subject:** Data Analytics — Global Marketing Management  

---

## 📌 Business Idea

**GlobalReach** is a cross-border digital marketing consultancy that helps brands expand into international markets through data-driven campaigns. This project uses synthetic data to validate GlobalReach's end-to-end sales pipeline — from lead generation to revenue and customer retention.

---

## 📁 Repository Structure

```
globalreach-analytics/
├── app.py                        # Streamlit EDA Dashboard (Step 3 — 30 Marks)
├── data_cleaning.py              # Data Cleaning & Transformation (Step 2 — 10 Marks)
├── requirements.txt              # Python dependencies
├── data/
│   └── globalreach_200rows.csv  # Synthetic Dataset (Step 1 — 10 Marks)
└── README.md
```

---

## 📊 Dataset Overview

| Attribute | Details |
|---|---|
| Rows | 200 |
| Columns | 24 (+ 7 derived after cleaning) |
| Date Range | Jan 2023 – Jun 2024 |
| Countries | 10 (USA, UK, Germany, UAE, Singapore, Australia, India, Brazil, Canada, France) |
| Industries | 7 (FMCG, Tech, Retail, Healthcare, Finance, Education, Hospitality) |

### Column Categories

- **Lead Generation**: Lead_ID, Date, Country, Region, Lead_Source
- **Campaign**: Campaign_Type, Ad_Spend_USD, Impressions, Clicks, CTR_Percent
- **Pipeline**: Lead_Score, Sales_Cycle_Days, Deal_Stage
- **Revenue**: Contract_Value_USD, Discount_Offered_Percent, Revenue_USD
- **Economics**: CAC_USD, CLV_USD
- **Retention**: NPS_Score, Churn

---

## 🧹 Data Cleaning Steps (Step 2)

1. **Missing Value Treatment** — Median imputation for Lead_Score, CTR, Discount, NPS (~5% nulls)
2. **Data Type Conversion** — Date to datetime; scores cast to int
3. **Outlier Handling** — IQR method on Ad_Spend, Revenue, CAC, CLV
4. **Derived Features** — ROI_Percent, CLV_CAC_Ratio, Lead_Conversion_Rate, Quarter, Is_Won
5. **Ordinal Encoding** — Company_Size and Deal_Stage encoded numerically
6. **Duplicate Removal** — Based on Lead_ID

---

## 📈 EDA Dashboard Features (Step 3)

- 📊 **Sales Funnel** — Prospect → Closed Won drop-off analysis
- 🌍 **Geo Analysis** — Revenue choropleth map, leads by region
- 📣 **Campaign Performance** — Ad Spend vs Revenue scatter, CTR by source, monthly trends
- 🔗 **Correlation Heatmap** — Pearson matrix across all numeric features
- 🔄 **Retention & Churn** — Churn by industry, NPS vs Churn overlay

---

## 🚀 Running the App Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/globalreach-analytics.git
cd globalreach-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App** → select your repo → set `app.py` as the main file
4. Click **Deploy** ✅

---

## 🔗 Tools Used

| Tool | Purpose |
|---|---|
| Python + Pandas | Data generation & cleaning |
| Plotly + Seaborn | Visualizations |
| Streamlit | Interactive dashboard |
| GitHub | Version control & deployment |

---

*Assignment submitted for Data Analytics subject — SP Jain School of Global Management*
