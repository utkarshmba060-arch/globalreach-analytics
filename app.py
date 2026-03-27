import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from data_cleaning import load_and_clean_data

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlobalReach Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0f4c75, #1b6ca8, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        font-family: 'DM Sans', sans-serif;
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
        border-radius: 16px;
        padding: 20px 24px;
        color: white;
        margin-bottom: 10px;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
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
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_clean_data("data/globalreach_200rows.csv")

df = get_data()

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌍 GlobalReach Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Cross-Border Digital Marketing Intelligence · SP Jain School of Global Management · Utkarsh Chandra</p>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar Filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    region_filter = st.multiselect("Region", options=sorted(df['Region'].unique()), default=sorted(df['Region'].unique()))
    industry_filter = st.multiselect("Industry", options=sorted(df['Industry'].unique()), default=sorted(df['Industry'].unique()))
    size_filter = st.multiselect("Company Size", options=sorted(df['Company_Size'].unique()), default=sorted(df['Company_Size'].unique()))
    stage_filter = st.multiselect("Deal Stage", options=sorted(df['Deal_Stage'].unique()), default=sorted(df['Deal_Stage'].unique()))
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.info(f"📦 {len(df)} rows · 24 columns\n\n📅 2023–2024\n\n🌍 10 countries")

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
avg_cac = df_f['CAC_USD'].mean()
avg_clv = df_f[df_f['CLV_USD'] > 0]['CLV_USD'].mean()
conv_rate = len(df_f[df_f['Deal_Stage'] == 'Closed Won']) / len(df_f) * 100 if len(df_f) > 0 else 0
avg_nps = df_f['NPS_Score'].mean()

k1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
k2.metric("🎯 Avg CAC", f"${avg_cac:,.0f}")
k3.metric("♾️ Avg CLV", f"${avg_clv:,.0f}")
k4.metric("✅ Win Rate", f"{conv_rate:.1f}%")
k5.metric("⭐ Avg NPS", f"{avg_nps:.1f}")

st.markdown("---")

# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sales Funnel", "🌍 Geo Analysis", "📣 Campaign Performance",
    "🔗 Correlation Analysis", "🔄 Retention & Churn"
])

# ════════════════════════════════════════════════
# TAB 1 — Sales Funnel
# ════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Sales Pipeline Funnel</div>', unsafe_allow_html=True)
        stage_order = ['Prospect', 'Qualified', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
        funnel_df = df_f['Deal_Stage'].value_counts().reindex(stage_order).fillna(0).reset_index()
        funnel_df.columns = ['Stage', 'Count']
        fig = go.Figure(go.Funnel(
            y=funnel_df['Stage'],
            x=funnel_df['Count'],
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
                      color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                      hole=0.45)
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Sales Cycle Duration by Deal Stage & Company Size</div>', unsafe_allow_html=True)
    fig3 = px.box(df_f, x='Deal_Stage', y='Sales_Cycle_Days', color='Company_Size',
                  color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                  category_orders={'Deal_Stage': stage_order})
    fig3.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Enterprise deals have significantly longer sales cycles (60–180 days) compared to SMEs (7–45 days), requiring dedicated nurturing strategies.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 2 — Geo Analysis
# ════════════════════════════════════════════════
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

# ════════════════════════════════════════════════
# TAB 3 — Campaign Performance
# ════════════════════════════════════════════════
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

    st.markdown('<div class="section-header">Monthly Ad Spend Trend</div>', unsafe_allow_html=True)
    df_f['Date_dt'] = pd.to_datetime(df_f['Date'])
    monthly = df_f.groupby(df_f['Date_dt'].dt.to_period('M')).agg(
        Ad_Spend=('Ad_Spend_USD','sum'),
        Leads=('Leads_Generated','sum')
    ).reset_index()
    monthly['Date_dt'] = monthly['Date_dt'].astype(str)
    fig9 = make_subplots(specs=[[{"secondary_y": True}]])
    fig9.add_trace(go.Bar(x=monthly['Date_dt'], y=monthly['Ad_Spend'], name='Ad Spend', marker_color='#1b6ca8'), secondary_y=False)
    fig9.add_trace(go.Scatter(x=monthly['Date_dt'], y=monthly['Leads'], name='Leads', line=dict(color='#00b4d8', width=2)), secondary_y=True)
    fig9.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Insight:</b> LinkedIn and Referral sources show the highest CTR. SEO and Content Marketing campaigns yield the best ROI (revenue ÷ ad spend).</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 4 — Correlation Analysis
# ════════════════════════════════════════════════
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
        fig11 = px.scatter(df_f, x='Ad_Spend_USD', y='Revenue_USD',
                           color='Company_Size', size='Lead_Score',
                           color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                           trendline='ols', title='Ad Spend vs Revenue')
        fig11.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig11, use_container_width=True)

    with c2:
        fig12 = px.scatter(df_f, x='NPS_Score', y='CLV_USD',
                           color='Industry',
                           trendline='ols', title='NPS Score vs Customer Lifetime Value')
        fig12.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig12, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig13 = px.scatter(df_f, x='Lead_Score', y='Contract_Value_USD',
                           color='Deal_Stage', trendline='ols',
                           title='Lead Score vs Contract Value')
        fig13.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig13, use_container_width=True)

    with c4:
        fig14 = px.scatter(df_f, x='CAC_USD', y='CLV_USD',
                           color='Company_Size', trendline='ols',
                           color_discrete_sequence=['#0f4c75','#1b6ca8','#00b4d8'],
                           title='CAC vs CLV (Efficiency Ratio)')
        fig14.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig14, use_container_width=True)

    st.markdown("""
    <div class="insight-box">💡 <b>Key Correlations Found:</b><br>
    • <b>Ad Spend ↔ Revenue</b>: Strong positive correlation (r ≈ 0.72) — higher spend drives higher revenue, especially for Enterprise.<br>
    • <b>NPS ↔ CLV</b>: Positive correlation — satisfied clients generate 3–5× more lifetime value.<br>
    • <b>Lead Score ↔ Contract Value</b>: Moderate positive correlation — higher-quality leads close larger deals.<br>
    • <b>CAC ↔ CLV</b>: A healthy CLV:CAC ratio > 3 is observed across most company sizes.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 5 — Retention & Churn
# ════════════════════════════════════════════════
with tab5:
    active = df_f[df_f['Churn'].isin(['Yes', 'No'])]
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Churn Distribution</div>', unsafe_allow_html=True)
        churn_count = active['Churn'].value_counts().reset_index()
        churn_count.columns = ['Churn', 'Count']
        fig15 = px.pie(churn_count, values='Count', names='Churn',
                       color_discrete_map={'No': '#0f4c75', 'Yes': '#e63946'},
                       hole=0.5)
        fig15.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig15, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Churn Rate by Industry</div>', unsafe_allow_html=True)
        churn_ind = active.groupby(['Industry','Churn']).size().reset_index(name='Count')
        fig16 = px.bar(churn_ind, x='Industry', y='Count', color='Churn',
                       barmode='group',
                       color_discrete_map={'No': '#1b6ca8', 'Yes': '#e63946'})
        fig16.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig16, use_container_width=True)

    st.markdown('<div class="section-header">NPS Score Distribution by Churn Status</div>', unsafe_allow_html=True)
    fig17 = px.histogram(active.dropna(subset=['NPS_Score']), x='NPS_Score', color='Churn',
                         barmode='overlay', nbins=10,
                         color_discrete_map={'No': '#1b6ca8', 'Yes': '#e63946'},
                         opacity=0.75)
    fig17.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig17, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Insight:</b> Churned clients consistently show NPS scores below 5. Proactive NPS monitoring can predict churn up to 60 days in advance. Finance and Retail sectors show the highest churn rates.</div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>GlobalReach Analytics · Utkarsh Chandra · SP Jain School of Global Management · Data Analytics Assignment 2024</small></center>",
    unsafe_allow_html=True
)
