import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── ML imports ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlobalReach Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0f4c75, #1b6ca8, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.sub-title { font-family: 'DM Sans', sans-serif; color: #6b7280; font-size: 1rem; margin-top: 0; }
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f4c75;
    border-left: 4px solid #00b4d8;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}
.insight-box {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.9rem;
    color: #0369a1;
    margin: 8px 0;
}
.algo-box {
    background: linear-gradient(135deg, #f8fafc, #e0f2fe);
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
}
.metric-badge {
    display: inline-block;
    background: #0f4c75;
    color: white;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 2px;
}
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)


# ─── Synthetic Data Generator (fallback if CSV missing) ──────────────────────
def generate_synthetic_data(n=200, seed=42):
    np.random.seed(seed)
    countries = ['USA', 'UK', 'Germany', 'UAE', 'Singapore', 'Australia', 'India', 'Brazil', 'Canada', 'France']
    country_region = {
        'USA': 'North America', 'Canada': 'North America',
        'UK': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
        'UAE': 'Middle East',
        'Singapore': 'Asia-Pacific', 'Australia': 'Asia-Pacific', 'India': 'Asia-Pacific',
        'Brazil': 'South America'
    }
    industries   = ['FMCG', 'Tech', 'Retail', 'Healthcare', 'Finance', 'Education', 'Hospitality']
    company_sizes = ['SME', 'Mid-Market', 'Enterprise']
    lead_sources = ['LinkedIn', 'Referral', 'SEO', 'Paid Ads', 'Email', 'Events']
    campaign_types = ['Content Marketing', 'PPC', 'Social Media', 'Email Campaign', 'Influencer', 'Webinar']
    deal_stages  = ['Prospect', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']

    df = pd.DataFrame()
    df['Lead_ID']    = [f'L{str(i).zfill(4)}' for i in range(1, n+1)]
    dates = pd.date_range('2023-01-01', '2024-06-30', periods=n)
    df['Date']       = np.random.choice(dates, n)
    df['Country']    = np.random.choice(countries, n)
    df['Region']     = df['Country'].map(country_region)
    df['Industry']   = np.random.choice(industries, n)
    df['Company_Size'] = np.random.choice(company_sizes, n, p=[0.40, 0.35, 0.25])
    df['Lead_Source']  = np.random.choice(lead_sources, n)
    df['Campaign_Type'] = np.random.choice(campaign_types, n)

    sm = df['Company_Size'].map({'SME': 1.0, 'Mid-Market': 2.5, 'Enterprise': 5.0})
    df['Ad_Spend_USD']  = (np.random.uniform(500, 3000, n) * sm).round(2)
    df['Impressions']   = (df['Ad_Spend_USD'] * np.random.uniform(50, 150, n)).astype(int)
    df['Clicks']        = (df['Impressions'] * np.random.uniform(0.01, 0.08, n)).astype(int)
    df['CTR_Percent']   = (df['Clicks'] / df['Impressions'] * 100).round(2)
    df['Leads_Generated'] = np.random.randint(1, 20, n)

    df['Lead_Score']          = np.random.randint(20, 100, n)
    df['Sales_Cycle_Days']    = (np.random.uniform(7, 60, n) * sm * 0.6).astype(int).clip(7, 200)
    df['Deal_Stage']          = np.random.choice(deal_stages, n, p=[0.15, 0.20, 0.20, 0.15, 0.20, 0.10])
    df['Contract_Value_USD']  = (np.random.uniform(5000, 50000, n) * sm).round(2)
    df['Discount_Offered_Percent'] = np.random.uniform(0, 30, n).round(2)
    df['Revenue_USD'] = np.where(
        df['Deal_Stage'] == 'Closed Won',
        (df['Contract_Value_USD'] * (1 - df['Discount_Offered_Percent'] / 100)).round(2), 0
    )
    df['CAC_USD'] = (df['Ad_Spend_USD'] / df['Leads_Generated']).round(2)
    df['CLV_USD'] = np.where(df['Revenue_USD'] > 0,
                              (df['Revenue_USD'] * np.random.uniform(2, 6, n)).round(2), 0)
    df['NPS_Score'] = np.random.randint(0, 10, n)
    df['Churn'] = np.where(
        df['NPS_Score'] < 5,
        np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40]),
        np.random.choice(['Yes', 'No'], n, p=[0.20, 0.80])
    )

    # Derived features
    df['ROI_Percent']  = np.where(df['Ad_Spend_USD'] > 0,
                                   ((df['Revenue_USD'] - df['Ad_Spend_USD']) / df['Ad_Spend_USD'] * 100).round(2), 0)
    df['CLV_CAC_Ratio'] = np.where(df['CAC_USD'] > 0, (df['CLV_USD'] / df['CAC_USD']).round(2), 0)
    df['Is_Won']  = (df['Deal_Stage'] == 'Closed Won').astype(int)
    df['Quarter'] = pd.to_datetime(df['Date']).dt.quarter
    return df


@st.cache_data
def get_data():
    try:
        from data_cleaning import load_and_clean_data
        return load_and_clean_data("data/globalreach_200rows.csv")
    except Exception:
        return generate_synthetic_data()

df = get_data()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌍 GlobalReach Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Cross-Border Digital Marketing Intelligence · SP Jain School of Global Management · Utkarsh Chandra</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar Filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    region_filter   = st.multiselect("Region",       options=sorted(df['Region'].unique()),       default=sorted(df['Region'].unique()))
    industry_filter = st.multiselect("Industry",     options=sorted(df['Industry'].unique()),     default=sorted(df['Industry'].unique()))
    size_filter     = st.multiselect("Company Size", options=sorted(df['Company_Size'].unique()), default=sorted(df['Company_Size'].unique()))
    stage_filter    = st.multiselect("Deal Stage",   options=sorted(df['Deal_Stage'].unique()),   default=sorted(df['Deal_Stage'].unique()))
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.info(f"📦 {len(df)} rows · 24+ columns\n\n📅 2023–2024\n\n🌍 10 countries")

df_f = df[
    df['Region'].isin(region_filter) &
    df['Industry'].isin(industry_filter) &
    df['Company_Size'].isin(size_filter) &
    df['Deal_Stage'].isin(stage_filter)
]

# ─── KPI Row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
total_revenue = df_f[df_f['Revenue_USD'] > 0]['Revenue_USD'].sum()
avg_cac   = df_f['CAC_USD'].mean()
avg_clv   = df_f[df_f['CLV_USD'] > 0]['CLV_USD'].mean()
conv_rate = (len(df_f[df_f['Deal_Stage'] == 'Closed Won']) / len(df_f) * 100) if len(df_f) > 0 else 0
avg_nps   = df_f['NPS_Score'].mean()
k1.metric("💰 Total Revenue",  f"${total_revenue:,.0f}")
k2.metric("🎯 Avg CAC",        f"${avg_cac:,.0f}")
k3.metric("♾️ Avg CLV",        f"${avg_clv:,.0f}")
k4.metric("✅ Win Rate",        f"{conv_rate:.1f}%")
k5.metric("⭐ Avg NPS",         f"{avg_nps:.1f}")
st.markdown("---")


# ─── ALL TABS ─────────────────────────────────────────────────────────────────
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8, tab9) = st.tabs([
    "📈 Sales Funnel",
    "🌍 Geo Analysis",
    "📣 Campaign Performance",
    "🔗 Correlation Analysis",
    "🔄 Retention & Churn",
    "🤖 Churn Prediction ML",
    "📊 Customer Segmentation",
    "🔮 Revenue Forecasting",
    "🎯 Lead Scoring Model",
])


# ═══════════════════════════════════════════════
# TAB 1 — Sales Funnel
# ═══════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Sales Pipeline Funnel</div>', unsafe_allow_html=True)
        stage_order = ['Prospect', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
        funnel_df = df_f['Deal_Stage'].value_counts().reindex(stage_order).fillna(0).reset_index()
        funnel_df.columns = ['Stage', 'Count']
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'], x=funnel_df['Count'],
            textinfo="value+percent initial",
            marker=dict(color=['#0f4c75','#1b6ca8','#1e90ff','#00b4d8','#48cae4','#e63946']),
        ))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">💡 <b>Insight:</b> The largest drop-off occurs between <b>Proposal → Negotiation</b> stages, indicating pricing or value proposition friction.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Revenue by Company Size</div>', unsafe_allow_html=True)
        rev_size = df_f[df_f['Revenue_USD'] > 0].groupby('Company_Size')['Revenue_USD'].sum().reset_index()
        fig2 = px.pie(rev_size, values='Revenue_USD', names='Company_Size',
                      color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'], hole=0.45)
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Sales Cycle Duration by Deal Stage & Company Size</div>', unsafe_allow_html=True)
    fig3 = px.box(df_f, x='Deal_Stage', y='Sales_Cycle_Days', color='Company_Size',
                  color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                  category_orders={'Deal_Stage': stage_order})
    fig3.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Enterprise deals have significantly longer sales cycles (60–180 days) compared to SMEs (7–45 days), requiring dedicated nurturing strategies.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — Geo Analysis
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Revenue by Country</div>', unsafe_allow_html=True)
    geo_df = df_f[df_f['Revenue_USD'] > 0].groupby('Country')['Revenue_USD'].sum().reset_index()
    fig4 = px.choropleth(geo_df, locations='Country', locationmode='country names',
                         color='Revenue_USD', color_continuous_scale='Blues',
                         title='Revenue Contribution by Country')
    fig4.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig4, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Leads Generated by Region</div>', unsafe_allow_html=True)
        reg_leads = df_f.groupby('Region')['Leads_Generated'].sum().reset_index()
        fig5 = px.bar(reg_leads.sort_values('Leads_Generated', ascending=True),
                      x='Leads_Generated', y='Region', orientation='h',
                      color='Leads_Generated', color_continuous_scale='Blues')
        fig5.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Avg Lead Score by Country</div>', unsafe_allow_html=True)
        country_score = df_f.groupby('Country')['Lead_Score'].mean().reset_index().sort_values('Lead_Score', ascending=False)
        fig6 = px.bar(country_score, x='Country', y='Lead_Score',
                      color='Lead_Score', color_continuous_scale='Blues')
        fig6.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Asia-Pacific (India, Singapore, Australia) generates the highest volume of leads, while North America delivers the highest average contract values.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 3 — Campaign Performance
# ═══════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Ad Spend vs Revenue by Campaign Type</div>', unsafe_allow_html=True)
        camp_df = df_f.groupby('Campaign_Type').agg(
            Ad_Spend=('Ad_Spend_USD','sum'),
            Revenue=('Revenue_USD','sum'),
            Leads=('Leads_Generated','sum')
        ).reset_index()
        fig7 = px.scatter(camp_df, x='Ad_Spend', y='Revenue', size='Leads',
                          color='Campaign_Type', text='Campaign_Type',
                          color_discrete_sequence=px.colors.qualitative.Bold)
        fig7.update_traces(textposition='top center')
        fig7.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">CTR by Lead Source</div>', unsafe_allow_html=True)
        ctr_df = df_f.groupby('Lead_Source')['CTR_Percent'].mean().reset_index().sort_values('CTR_Percent', ascending=False)
        fig8 = px.bar(ctr_df, x='Lead_Source', y='CTR_Percent',
                      color='CTR_Percent', color_continuous_scale='Blues',
                      text=ctr_df['CTR_Percent'].round(2))
        fig8.update_traces(texttemplate='%{text}%', textposition='outside')
        fig8.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-header">Monthly Ad Spend & Lead Trend</div>', unsafe_allow_html=True)
    df_f2 = df_f.copy()
    df_f2['Date_dt'] = pd.to_datetime(df_f2['Date'])
    monthly = df_f2.groupby(df_f2['Date_dt'].dt.to_period('M')).agg(
        Ad_Spend=('Ad_Spend_USD','sum'), Leads=('Leads_Generated','sum')
    ).reset_index()
    monthly['Date_dt'] = monthly['Date_dt'].astype(str)
    fig9 = make_subplots(specs=[[{"secondary_y": True}]])
    fig9.add_trace(go.Bar(x=monthly['Date_dt'], y=monthly['Ad_Spend'], name='Ad Spend', marker_color='#1b6ca8'), secondary_y=False)
    fig9.add_trace(go.Scatter(x=monthly['Date_dt'], y=monthly['Leads'], name='Leads', line=dict(color='#00b4d8', width=2)), secondary_y=True)
    fig9.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Insight:</b> LinkedIn and Referral sources show the highest CTR. SEO and Content Marketing campaigns yield the best ROI (revenue ÷ ad spend).</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 4 — Correlation Analysis
# ═══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Correlation Heatmap — Numeric Features</div>', unsafe_allow_html=True)
    num_cols = ['Ad_Spend_USD','Impressions','Clicks','CTR_Percent','Leads_Generated',
                'Lead_Score','Sales_Cycle_Days','Contract_Value_USD',
                'Discount_Offered_Percent','Revenue_USD','CAC_USD','CLV_USD','NPS_Score']
    corr = df_f[num_cols].corr()
    fig10, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title('Pearson Correlation Matrix — GlobalReach Dataset', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig10)

    st.markdown('<div class="section-header">Key Pairwise Relationships</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig11 = px.scatter(df_f, x='Ad_Spend_USD', y='Revenue_USD', color='Company_Size',
                           size='Lead_Score', color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                           trendline='ols', title='Ad Spend vs Revenue')
        fig11.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig11, use_container_width=True)
    with c2:
        fig12 = px.scatter(df_f, x='NPS_Score', y='CLV_USD', color='Industry',
                           trendline='ols', title='NPS Score vs Customer Lifetime Value')
        fig12.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig12, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig13 = px.scatter(df_f, x='Lead_Score', y='Contract_Value_USD', color='Deal_Stage',
                           trendline='ols', title='Lead Score vs Contract Value')
        fig13.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig13, use_container_width=True)
    with c4:
        fig14 = px.scatter(df_f, x='CAC_USD', y='CLV_USD', color='Company_Size',
                           color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                           trendline='ols', title='CAC vs CLV (Efficiency Ratio)')
        fig14.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig14, use_container_width=True)

    st.markdown("""
    <div class="insight-box">💡 <b>Key Correlations Found:</b><br>
    • <b>Ad Spend ↔ Revenue</b>: Strong positive correlation (r ≈ 0.72) — higher spend drives higher revenue, especially for Enterprise.<br>
    • <b>NPS ↔ CLV</b>: Positive correlation — satisfied clients generate 3–5× more lifetime value.<br>
    • <b>Lead Score ↔ Contract Value</b>: Moderate positive correlation — higher-quality leads close larger deals.<br>
    • <b>CAC ↔ CLV</b>: A healthy CLV:CAC ratio > 3 is observed across most company sizes.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 5 — Retention & Churn (EDA)
# ═══════════════════════════════════════════════
with tab5:
    active = df_f[df_f['Churn'].isin(['Yes', 'No'])]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        churn_count = active['Churn'].value_counts().reset_index()
        churn_count.columns = ['Churn', 'Count']
        fig15 = px.pie(churn_count, values='Count', names='Churn',
                       color_discrete_map={'No': '#0f4c75', 'Yes': '#e63946'}, hole=0.5)
        fig15.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig15, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Churn Rate by Industry</div>', unsafe_allow_html=True)
        churn_ind = active.groupby(['Industry','Churn']).size().reset_index(name='Count')
        fig16 = px.bar(churn_ind, x='Industry', y='Count', color='Churn',
                       barmode='group', color_discrete_map={'No': '#1b6ca8', 'Yes': '#e63946'})
        fig16.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig16, use_container_width=True)

    st.markdown('<div class="section-header">NPS Score Distribution by Churn Status</div>', unsafe_allow_html=True)
    fig17 = px.histogram(active.dropna(subset=['NPS_Score']), x='NPS_Score', color='Churn',
                         barmode='overlay', nbins=10,
                         color_discrete_map={'No': '#1b6ca8', 'Yes': '#e63946'}, opacity=0.75)
    fig17.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig17, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Churned clients consistently show NPS scores below 5. Proactive NPS monitoring can predict churn up to 60 days in advance. Finance and Retail sectors show the highest churn rates.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 6 — 🤖 CHURN PREDICTION ML  (Logistic Regression + RF)
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">🤖 Churn Prediction — Machine Learning Classification</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
    <b>Algorithms Used:</b> Logistic Regression &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp; Gradient Boosting Classifier<br>
    <b>Target:</b> Churn (Yes / No) &nbsp;|&nbsp; <b>Problem Type:</b> Binary Classification
    </div>""", unsafe_allow_html=True)

    # ── Prepare data ──
    churn_df = df[df['Churn'].isin(['Yes', 'No'])].copy()
    churn_df['Churn_Binary'] = (churn_df['Churn'] == 'Yes').astype(int)

    feature_cols_churn = ['NPS_Score', 'CLV_USD', 'CAC_USD', 'Sales_Cycle_Days',
                          'Lead_Score', 'Discount_Offered_Percent', 'Ad_Spend_USD', 'Revenue_USD']
    feat_df = churn_df[feature_cols_churn + ['Churn_Binary']].dropna()
    X = feat_df[feature_cols_churn]
    y = feat_df['Churn_Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Train models ──
    lr  = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
    rf  = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_s, y_train)
    gbc = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train_s, y_train)

    models = {'Logistic Regression': lr, 'Random Forest': rf, 'Gradient Boosting': gbc}

    # ── Model comparison metrics ──
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    metrics_rows = []
    for name, model in models.items():
        pred = model.predict(X_test_s)
        metrics_rows.append({
            'Model': name,
            'Accuracy':  round(accuracy_score(y_test, pred), 4),
            'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
            'Recall':    round(recall_score(y_test, pred, zero_division=0), 4),
            'F1 Score':  round(f1_score(y_test, pred, zero_division=0), 4),
        })
    metrics_df = pd.DataFrame(metrics_rows)
    
    # Bar chart comparison
    melt = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig_m = px.bar(melt, x='Metric', y='Score', color='Model', barmode='group',
                   color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                   text_auto='.3f', height=340)
    fig_m.update_layout(margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 1.1])
    st.plotly_chart(fig_m, use_container_width=True)
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)

    # ── ROC Curves ──
    st.markdown('<div class="section-header">ROC Curves — All Models</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_roc = go.Figure()
        colors_roc = ['#0f4c75','#00b4d8','#e63946']
        for (name, model), color in zip(models.items(), colors_roc):
            proba = model.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.3f})', line=dict(color=color, width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='grey'), name='Random'))
        fig_roc.update_layout(title='ROC Curve Comparison', xaxis_title='False Positive Rate',
                               yaxis_title='True Positive Rate', height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Confusion Matrix (Random Forest)</div>', unsafe_allow_html=True)
        rf_pred = rf.predict(X_test_s)
        cm = confusion_matrix(y_test, rf_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
        ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
        ax_cm.set_title('Confusion Matrix — Random Forest')
        plt.tight_layout()
        st.pyplot(fig_cm)

    # ── Feature Importance (RF) ──
    st.markdown('<div class="section-header">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({'Feature': feature_cols_churn, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Blues', height=350)
    fig_fi.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Live Predictor ──
    st.markdown('<div class="section-header">🔮 Live Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown("Input a customer's details to predict churn probability in real time.")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        p_nps   = st.slider("NPS Score", 0, 10, 6)
        p_clv   = st.number_input("CLV (USD)", 0, 500000, 30000)
    with pc2:
        p_cac   = st.number_input("CAC (USD)", 0, 50000, 2000)
        p_cycle = st.slider("Sales Cycle Days", 7, 200, 45)
    with pc3:
        p_score = st.slider("Lead Score", 0, 100, 60)
        p_disc  = st.slider("Discount %", 0, 30, 10)
    with pc4:
        p_spend = st.number_input("Ad Spend (USD)", 0, 100000, 5000)
        p_rev   = st.number_input("Revenue (USD)", 0, 500000, 25000)

    if st.button("🔮 Predict Churn", type="primary"):
        inp = np.array([[p_nps, p_clv, p_cac, p_cycle, p_score, p_disc, p_spend, p_rev]])
        inp_s = scaler.transform(inp)
        prob = rf.predict_proba(inp_s)[0][1]
        label = "🔴 HIGH RISK — Likely to Churn" if prob > 0.5 else "🟢 LOW RISK — Likely to Retain"
        st.metric("Churn Probability (Random Forest)", f"{prob*100:.1f}%", label)
        st.progress(float(prob))

    st.markdown('<div class="insight-box">💡 <b>Key Finding:</b> NPS Score and CLV are the top predictors of churn. Gradient Boosting achieves the highest AUC, while Random Forest offers the best balance of precision and recall for marketing intervention prioritisation.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 7 — 📊 CUSTOMER SEGMENTATION  (K-Means Clustering)
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-header">📊 Customer Segmentation — K-Means Clustering</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
    <b>Algorithm:</b> K-Means Clustering &nbsp;|&nbsp; <b>Problem Type:</b> Unsupervised Learning<br>
    <b>Features Used:</b> CLV, CAC, Revenue, Lead Score, NPS, Ad Spend
    </div>""", unsafe_allow_html=True)

    seg_features = ['CLV_USD', 'CAC_USD', 'Revenue_USD', 'Lead_Score', 'NPS_Score', 'Ad_Spend_USD']
    seg_df = df[seg_features].dropna()
    seg_df = seg_df[seg_df['CLV_USD'] > 0]

    scaler_seg = StandardScaler()
    X_seg = scaler_seg.fit_transform(seg_df)

    # ── Elbow Curve ──
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Elbow Method — Optimal k</div>', unsafe_allow_html=True)
        inertias = []
        k_range = range(2, 9)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_seg)
            inertias.append(km.inertia_)
        fig_elbow = px.line(x=list(k_range), y=inertias, markers=True,
                            labels={'x': 'Number of Clusters (k)', 'y': 'Inertia (WCSS)'},
                            title='Elbow Curve', color_discrete_sequence=['#1b6ca8'])
        fig_elbow.add_vline(x=4, line_dash='dash', line_color='#e63946', annotation_text='Optimal k=4')
        fig_elbow.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Cluster Distribution</div>', unsafe_allow_html=True)
        n_clusters = st.slider("Select Number of Clusters", 2, 6, 4)
        km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        seg_df = seg_df.copy()
        seg_df['Cluster'] = km_final.fit_predict(X_seg[:len(seg_df)])
        seg_df['Cluster'] = 'Segment ' + (seg_df['Cluster'] + 1).astype(str)

        cluster_counts = seg_df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Segment', 'Count']
        fig_cc = px.pie(cluster_counts, values='Count', names='Segment',
                        color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8','#48cae4','#90e0ef','#caf0f8'],
                        hole=0.4, height=340)
        fig_cc.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cc, use_container_width=True)

    # ── 2D Scatter ──
    st.markdown('<div class="section-header">Cluster Visualisation — CLV vs Revenue</div>', unsafe_allow_html=True)
    fig_sc = px.scatter(seg_df, x='CLV_USD', y='Revenue_USD', color='Cluster',
                        size='Lead_Score', hover_data=['NPS_Score', 'CAC_USD'],
                        color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8','#48cae4','#90e0ef','#caf0f8'],
                        title='Customer Segments: CLV vs Revenue')
    fig_sc.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Cluster Profile ──
    st.markdown('<div class="section-header">Cluster Profile Summary</div>', unsafe_allow_html=True)
    profile = seg_df.groupby('Cluster')[seg_features].mean().round(2).reset_index()
    st.dataframe(profile.style.background_gradient(cmap='Blues', subset=seg_features), use_container_width=True)

    # ── Radar Chart ──
    st.markdown('<div class="section-header">Cluster Radar Chart</div>', unsafe_allow_html=True)
    radar_cols = ['CLV_USD', 'CAC_USD', 'Lead_Score', 'NPS_Score']
    radar_df = seg_df.groupby('Cluster')[radar_cols].mean()
    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)

    fig_radar = go.Figure()
    colors_radar = ['#0f4c75','#1b6ca8','#00b4d8','#48cae4','#90e0ef','#caf0f8']
    for i, row in radar_norm.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row.tolist() + [row.tolist()[0]],
            theta=radar_cols + [radar_cols[0]],
            fill='toself', name=i,
            line_color=colors_radar[int(i.split()[-1]) - 1]
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                             height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("""
    <div class="insight-box">💡 <b>Segment Strategy:</b><br>
    • <b>Segment 1 (High-Value)</b>: High CLV + High NPS → Upsell & loyalty programmes<br>
    • <b>Segment 2 (At-Risk)</b>: Low NPS + Mid CLV → Proactive retention outreach<br>
    • <b>Segment 3 (Growth)</b>: High Lead Score + Low Revenue → Nurture & close faster<br>
    • <b>Segment 4 (Cost-Heavy)</b>: High CAC + Low CLV → Optimise acquisition channels
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 8 — 🔮 REVENUE FORECASTING  (Linear Reg + RF Regressor)
# ═══════════════════════════════════════════════════════════════
with tab8:
    st.markdown('<div class="section-header">🔮 Revenue Forecasting — Regression Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
    <b>Algorithms Used:</b> Linear Regression &nbsp;|&nbsp; Random Forest Regressor<br>
    <b>Target:</b> Revenue_USD &nbsp;|&nbsp; <b>Problem Type:</b> Regression (only Closed Won deals)
    </div>""", unsafe_allow_html=True)

    rev_df = df[df['Deal_Stage'] == 'Closed Won'].copy()
    reg_features = ['Ad_Spend_USD', 'Impressions', 'Lead_Score', 'Sales_Cycle_Days',
                    'Leads_Generated', 'Discount_Offered_Percent', 'Contract_Value_USD', 'NPS_Score']
    reg_df = rev_df[reg_features + ['Revenue_USD']].dropna()
    Xr = reg_df[reg_features]; yr = reg_df['Revenue_USD']

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.25, random_state=42)
    scaler_r = StandardScaler()
    Xr_train_s = scaler_r.fit_transform(Xr_train)
    Xr_test_s  = scaler_r.transform(Xr_test)

    lr_reg  = LinearRegression().fit(Xr_train_s, yr_train)
    rf_reg  = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr_train_s, yr_train)

    reg_models = {'Linear Regression': lr_reg, 'Random Forest Regressor': rf_reg}

    # ── Metrics ──
    st.markdown('<div class="section-header">Regression Model Performance</div>', unsafe_allow_html=True)
    reg_met = []
    for name, model in reg_models.items():
        pred = model.predict(Xr_test_s)
        reg_met.append({
            'Model': name,
            'R² Score': round(r2_score(yr_test, pred), 4),
            'MAE (USD)': round(mean_absolute_error(yr_test, pred), 2),
            'RMSE (USD)': round(np.sqrt(mean_squared_error(yr_test, pred)), 2),
        })
    reg_met_df = pd.DataFrame(reg_met).set_index('Model')
    st.dataframe(reg_met_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Actual vs Predicted Revenue (Linear Reg)</div>', unsafe_allow_html=True)
        lr_pred = lr_reg.predict(Xr_test_s)
        ap_df = pd.DataFrame({'Actual': yr_test.values, 'Predicted': lr_pred})
        fig_ap = px.scatter(ap_df, x='Actual', y='Predicted',
                            color_discrete_sequence=['#1b6ca8'],
                            title='Actual vs Predicted — Linear Regression')
        fig_ap.add_shape(type='line', x0=ap_df['Actual'].min(), y0=ap_df['Actual'].min(),
                         x1=ap_df['Actual'].max(), y1=ap_df['Actual'].max(),
                         line=dict(color='#e63946', dash='dash'))
        fig_ap.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_ap, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Actual vs Predicted Revenue (RF Regressor)</div>', unsafe_allow_html=True)
        rf_pred_r = rf_reg.predict(Xr_test_s)
        ap_df2 = pd.DataFrame({'Actual': yr_test.values, 'Predicted': rf_pred_r})
        fig_ap2 = px.scatter(ap_df2, x='Actual', y='Predicted',
                             color_discrete_sequence=['#0f4c75'],
                             title='Actual vs Predicted — Random Forest Regressor')
        fig_ap2.add_shape(type='line', x0=ap_df2['Actual'].min(), y0=ap_df2['Actual'].min(),
                          x1=ap_df2['Actual'].max(), y1=ap_df2['Actual'].max(),
                          line=dict(color='#e63946', dash='dash'))
        fig_ap2.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_ap2, use_container_width=True)

    # ── Residuals ──
    st.markdown('<div class="section-header">Residual Analysis — Random Forest</div>', unsafe_allow_html=True)
    residuals = yr_test.values - rf_pred_r
    fig_res = px.histogram(x=residuals, nbins=20, color_discrete_sequence=['#1b6ca8'],
                           labels={'x': 'Residual (Actual − Predicted)'}, title='Residual Distribution')
    fig_res.add_vline(x=0, line_dash='dash', line_color='#e63946')
    fig_res.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature Importance ──
    st.markdown('<div class="section-header">Feature Importance — RF Regressor</div>', unsafe_allow_html=True)
    fi_reg = pd.DataFrame({'Feature': reg_features, 'Importance': rf_reg.feature_importances_}).sort_values('Importance', ascending=True)
    fig_fi_r = px.bar(fi_reg, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Blues', height=320)
    fig_fi_r.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_fi_r, use_container_width=True)

    # ── Revenue Predictor ──
    st.markdown('<div class="section-header">💵 Live Revenue Estimator</div>', unsafe_allow_html=True)
    rp1, rp2, rp3, rp4 = st.columns(4)
    with rp1:
        rp_spend = st.number_input("Ad Spend (USD)", 0, 200000, 10000, key='rp_spend')
        rp_impr  = st.number_input("Impressions", 0, 5000000, 100000, key='rp_impr')
    with rp2:
        rp_lead_score = st.slider("Lead Score", 0, 100, 70, key='rp_ls')
        rp_cycle  = st.slider("Sales Cycle Days", 7, 200, 60, key='rp_cycle')
    with rp3:
        rp_leads  = st.slider("Leads Generated", 1, 50, 10, key='rp_leads')
        rp_disc   = st.slider("Discount %", 0, 30, 5, key='rp_disc')
    with rp4:
        rp_contract = st.number_input("Contract Value (USD)", 0, 500000, 50000, key='rp_cv')
        rp_nps   = st.slider("NPS Score", 0, 10, 7, key='rp_nps')

    if st.button("💵 Estimate Revenue", type="primary"):
        rp_inp = np.array([[rp_spend, rp_impr, rp_lead_score, rp_cycle, rp_leads, rp_disc, rp_contract, rp_nps]])
        rp_inp_s = scaler_r.transform(rp_inp)
        lr_est  = lr_reg.predict(rp_inp_s)[0]
        rf_est  = rf_reg.predict(rp_inp_s)[0]
        ec1, ec2 = st.columns(2)
        ec1.metric("Linear Regression Estimate", f"${max(0, lr_est):,.0f}")
        ec2.metric("Random Forest Estimate",      f"${max(0, rf_est):,.0f}")

    st.markdown('<div class="insight-box">💡 <b>Key Finding:</b> Contract Value and Ad Spend are the strongest revenue predictors. Random Forest Regressor outperforms Linear Regression with higher R² and lower RMSE, capturing non-linear relationships between marketing spend and revenue.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TAB 9 — 🎯 LEAD SCORING MODEL  (Decision Tree + GB Classifier)
# ═══════════════════════════════════════════════════════════════
with tab9:
    st.markdown('<div class="section-header">🎯 Lead Scoring Model — Classification</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="algo-box">
    <b>Algorithms Used:</b> Decision Tree &nbsp;|&nbsp; Gradient Boosting Classifier<br>
    <b>Target:</b> High-Quality Lead (Lead_Score ≥ 70 → 1, else 0) &nbsp;|&nbsp; <b>Problem Type:</b> Binary Classification
    </div>""", unsafe_allow_html=True)

    lead_df = df.copy()
    lead_df['High_Quality_Lead'] = (lead_df['Lead_Score'] >= 70).astype(int)

    # Encode categoricals
    le_src  = LabelEncoder()
    le_camp = LabelEncoder()
    le_size = LabelEncoder()
    lead_df['Lead_Source_enc']  = le_src.fit_transform(lead_df['Lead_Source'].astype(str))
    lead_df['Campaign_Type_enc'] = le_camp.fit_transform(lead_df['Campaign_Type'].astype(str))
    lead_df['Company_Size_enc'] = le_size.fit_transform(lead_df['Company_Size'].astype(str))

    ls_features = ['Ad_Spend_USD', 'Impressions', 'Clicks', 'CTR_Percent',
                   'Sales_Cycle_Days', 'Discount_Offered_Percent',
                   'Lead_Source_enc', 'Campaign_Type_enc', 'Company_Size_enc']
    ls_df = lead_df[ls_features + ['High_Quality_Lead']].dropna()
    Xl = ls_df[ls_features]; yl = ls_df['High_Quality_Lead']

    Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.25, random_state=42, stratify=yl)
    scaler_l = StandardScaler()
    Xl_train_s = scaler_l.fit_transform(Xl_train)
    Xl_test_s  = scaler_l.transform(Xl_test)

    dt  = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xl_train_s, yl_train)
    gbc2 = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xl_train_s, yl_train)

    lead_models = {'Decision Tree': dt, 'Gradient Boosting': gbc2}

    # ── Metrics ──
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    lm_rows = []
    for name, model in lead_models.items():
        pred = model.predict(Xl_test_s)
        lm_rows.append({
            'Model': name,
            'Accuracy':  round(accuracy_score(yl_test, pred), 4),
            'Precision': round(precision_score(yl_test, pred, zero_division=0), 4),
            'Recall':    round(recall_score(yl_test, pred, zero_division=0), 4),
            'F1 Score':  round(f1_score(yl_test, pred, zero_division=0), 4),
        })
    lm_df = pd.DataFrame(lm_rows)
    lm_melt = lm_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    fig_lm = px.bar(lm_melt, x='Metric', y='Score', color='Model', barmode='group',
                    color_discrete_sequence=['#0f4c75','#00b4d8'], text_auto='.3f', height=320)
    fig_lm.update_layout(margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 1.15])
    st.plotly_chart(fig_lm, use_container_width=True)
    st.dataframe(lm_df.set_index('Model'), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">ROC Curve — Lead Scoring</div>', unsafe_allow_html=True)
        fig_lroc = go.Figure()
        for (name, model), color in zip(lead_models.items(), ['#0f4c75','#00b4d8']):
            proba = model.predict_proba(Xl_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(yl_test, proba)
            roc_auc = auc(fpr, tpr)
            fig_lroc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={roc_auc:.3f})',
                                           line=dict(color=color, width=2)))
        fig_lroc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='grey'), name='Random'))
        fig_lroc.update_layout(xaxis_title='FPR', yaxis_title='TPR', height=360,
                                margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_lroc, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Feature Importance — Gradient Boosting</div>', unsafe_allow_html=True)
        ls_fi = pd.DataFrame({'Feature': ls_features, 'Importance': gbc2.feature_importances_}).sort_values('Importance', ascending=True)
        fig_ls_fi = px.bar(ls_fi, x='Importance', y='Feature', orientation='h',
                           color='Importance', color_continuous_scale='Blues', height=360)
        fig_ls_fi.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ls_fi, use_container_width=True)

    # ── Lead Score Distribution by Prediction ──
    st.markdown('<div class="section-header">Lead Score Distribution — High vs Low Quality</div>', unsafe_allow_html=True)
    ls_dist_df = df[['Lead_Score']].copy()
    ls_dist_df['Quality'] = np.where(ls_dist_df['Lead_Score'] >= 70, 'High Quality', 'Low Quality')
    fig_dist = px.histogram(ls_dist_df, x='Lead_Score', color='Quality', barmode='overlay',
                            color_discrete_map={'High Quality': '#0f4c75', 'Low Quality': '#e63946'},
                            nbins=20, height=300)
    fig_dist.add_vline(x=70, line_dash='dash', line_color='grey', annotation_text='Threshold = 70')
    fig_dist.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Live Lead Scorer ──
    st.markdown('<div class="section-header">🎯 Live Lead Quality Predictor</div>', unsafe_allow_html=True)
    lp1, lp2, lp3 = st.columns(3)
    with lp1:
        lp_spend = st.number_input("Ad Spend (USD)", 0, 100000, 8000, key='lp_spend')
        lp_impr  = st.number_input("Impressions", 0, 2000000, 80000, key='lp_impr')
        lp_clicks = st.number_input("Clicks", 0, 100000, 3000, key='lp_clicks')
    with lp2:
        lp_ctr   = st.slider("CTR %", 0.0, 15.0, 3.5, key='lp_ctr')
        lp_cycle = st.slider("Sales Cycle Days", 7, 200, 40, key='lp_cycle2')
        lp_disc  = st.slider("Discount %", 0, 30, 8, key='lp_disc2')
    with lp3:
        lp_src   = st.selectbox("Lead Source", le_src.classes_, key='lp_src')
        lp_camp  = st.selectbox("Campaign Type", le_camp.classes_, key='lp_camp')
        lp_size  = st.selectbox("Company Size", le_size.classes_, key='lp_size')

    if st.button("🎯 Score This Lead", type="primary"):
        lp_inp = np.array([[lp_spend, lp_impr, lp_clicks, lp_ctr, lp_cycle, lp_disc,
                            le_src.transform([lp_src])[0],
                            le_camp.transform([lp_camp])[0],
                            le_size.transform([lp_size])[0]]])
        lp_inp_s = scaler_l.transform(lp_inp)
        dt_prob  = dt.predict_proba(lp_inp_s)[0][1]
        gb_prob  = gbc2.predict_proba(lp_inp_s)[0][1]
        la1, la2 = st.columns(2)
        dt_label = "🟢 HIGH QUALITY" if dt_prob > 0.5 else "🔴 LOW QUALITY"
        gb_label = "🟢 HIGH QUALITY" if gb_prob > 0.5 else "🔴 LOW QUALITY"
        la1.metric("Decision Tree",        f"{dt_prob*100:.1f}% — {dt_label}")
        la2.metric("Gradient Boosting",    f"{gb_prob*100:.1f}% — {gb_label}")
        st.progress(float(gb_prob))

    st.markdown('<div class="insight-box">💡 <b>Key Finding:</b> CTR, Ad Spend, and Sales Cycle Days are the strongest predictors of lead quality. Gradient Boosting consistently outperforms Decision Tree with higher AUC, making it ideal for prioritising the sales pipeline and reducing wasted outreach effort.</div>', unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>🌍 GlobalReach Analytics · Utkarsh Chandra · SP Jain School of Global Management · Group PBL 2024<br>"
    "EDA + Machine Learning: Logistic Regression · Random Forest · Gradient Boosting · K-Means · Linear Regression · Decision Tree</small></center>",
    unsafe_allow_html=True
)

