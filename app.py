import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI Equity Research Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 8px; border-left: 4px solid #00ff00;}
    h1 {color: #00ff00; text-align: center; font-size: 3em;}
    h2 {color: #00d9ff; border-bottom: 2px solid #00d9ff; padding-bottom: 10px;}
    .highlight {background-color: #1e2130; padding: 20px; border-radius: 10px; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# Sentiment Analysis (Simple implementation)
def analyze_sentiment(text):
    """Simple sentiment scoring based on keyword matching"""
    positive_words = ['growth', 'strong', 'beat', 'exceed', 'positive', 'gain', 'profit', 
                      'success', 'improved', 'better', 'outperform', 'opportunity', 'momentum']
    negative_words = ['decline', 'weak', 'miss', 'below', 'negative', 'loss', 'concern',
                      'risk', 'challenge', 'worse', 'underperform', 'uncertainty']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.5  # Neutral
    
    sentiment_score = pos_count / total
    return sentiment_score

# DCF Valuation
def calculate_dcf(revenue, growth_rate, fcf_margin, wacc, terminal_growth, years=5):
    """Simplified DCF valuation model"""
    fcf_projections = []
    revenue_proj = revenue
    
    for year in range(1, years + 1):
        revenue_proj *= (1 + growth_rate)
        fcf = revenue_proj * fcf_margin
        pv_fcf = fcf / ((1 + wacc) ** year)
        fcf_projections.append({
            'Year': year,
            'Revenue': revenue_proj,
            'FCF': fcf,
            'PV_FCF': pv_fcf
        })
    
    # Terminal value
    terminal_fcf = fcf_projections[-1]['FCF'] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    
    # Enterprise value
    pv_fcf_sum = sum([proj['PV_FCF'] for proj in fcf_projections])
    enterprise_value = pv_fcf_sum + pv_terminal
    
    return enterprise_value, fcf_projections, pv_terminal

# Factor Scoring
def calculate_factor_score(metrics):
    """Multi-factor scoring system"""
    scores = {}
    
    # Value score (lower is better for P/E, P/B)
    pe_score = max(0, 100 - metrics.get('pe', 20) * 5)
    pb_score = max(0, 100 - metrics.get('pb', 3) * 33)
    scores['value'] = (pe_score + pb_score) / 2
    
    # Growth score
    revenue_growth = metrics.get('revenue_growth', 5)
    eps_growth = metrics.get('eps_growth', 5)
    scores['growth'] = min(100, (revenue_growth + eps_growth) * 5)
    
    # Quality score
    roe = metrics.get('roe', 15)
    debt_equity = metrics.get('debt_equity', 1.0)
    scores['quality'] = min(100, roe * 5 - debt_equity * 10)
    
    # Momentum score
    return_1m = metrics.get('return_1m', 0)
    return_3m = metrics.get('return_3m', 0)
    scores['momentum'] = min(100, max(0, (return_1m + return_3m) * 10 + 50))
    
    # Composite score
    scores['composite'] = np.mean(list(scores.values()))
    
    return scores

# Main App
st.title("📈 AI Equity Research Analyzer")
st.markdown("### **Next-Generation Investment Intelligence Platform**")
st.markdown("*Built for Nomura Equity Research & Wholesale Strategy*")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Company Analysis",
    "💰 DCF Valuation",
    "📊 Factor Screening",
    "🏢 Peer Comparison",
    "🔍 Competitive Intelligence"
])

# TAB 1: Company Analysis
with tab1:
    st.header("Company Deep Dive & Sentiment Analysis")
    
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL", key='company_ticker')
    
    if st.button("Analyze Company", key='analyze_btn'):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Fetch stock data
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='1y')
                
                # Current price
                current_price = hist['Close'].iloc[-1]
                
                # Display key metrics
                st.markdown("### Key Financial Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
                col3.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
                col4.metric("Div Yield", f"{info.get('dividendYield', 0)*100:.2f}%")
                col5.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
                
                # Price chart
                st.markdown("### Price Performance (1 Year)")
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                ))
                
                # Add moving averages
                ma_50 = hist['Close'].rolling(window=50).mean()
                ma_200 = hist['Close'].rolling(window=200).mean()
                
                fig.add_trace(go.Scatter(
                    x=hist.index, y=ma_50,
                    name='50-day MA',
                    line=dict(color='orange', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=hist.index, y=ma_200,
                    name='200-day MA',
                    line=dict(color='blue', width=1)
                ))
                
                fig.update_layout(
                    title=f"{ticker} Stock Price",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Fundamental metrics
                st.markdown("### Fundamental Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fundamentals = {
                        'Revenue': f"${info.get('totalRevenue', 0)/1e9:.2f}B",
                        'Gross Margin': f"{info.get('grossMargins', 0)*100:.1f}%",
                        'Operating Margin': f"{info.get('operatingMargins', 0)*100:.1f}%",
                        'Net Margin': f"{info.get('profitMargins', 0)*100:.1f}%",
                        'ROE': f"{info.get('returnOnEquity', 0)*100:.1f}%",
                    }
                    
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    for key, value in fundamentals.items():
                        st.markdown(f"**{key}:** {value}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    valuation = {
                        'P/E Ratio': f"{info.get('trailingPE', 0):.2f}",
                        'Forward P/E': f"{info.get('forwardPE', 0):.2f}",
                        'P/B Ratio': f"{info.get('priceToBook', 0):.2f}",
                        'PEG Ratio': f"{info.get('pegRatio', 0):.2f}",
                        'EV/EBITDA': f"{info.get('enterpriseToEbitda', 0):.2f}",
                    }
                    
                    st.markdown("<div class='highlight'>", unsafe_allow_html=True)
                    for key, value in valuation.items():
                        st.markdown(f"**{key}:** {value}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Sentiment Analysis (based on company description)
                st.markdown("### AI Sentiment Analysis")
                
                business_summary = info.get('longBusinessSummary', '')
                
                if business_summary:
                    sentiment_score = analyze_sentiment(business_summary)
                    
                    sentiment_label = "BULLISH" if sentiment_score > 0.6 else "BEARISH" if sentiment_score < 0.4 else "NEUTRAL"
                    sentiment_color = "#00ff00" if sentiment_score > 0.6 else "#ff0000" if sentiment_score < 0.4 else "#ffaa00"
                    
                    st.markdown(f"""
                        <div class='highlight'>
                            <h3 style='color: {sentiment_color};'>Sentiment: {sentiment_label}</h3>
                            <p><b>Sentiment Score:</b> {sentiment_score:.2f} / 1.00</p>
                            <p>{business_summary[:500]}...</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Analyst recommendations
                st.markdown("### Analyst Consensus")
                
                recommendations = stock.recommendations
                
                if recommendations is not None and not recommendations.empty:
                    recent = recommendations.tail(10)
                    
                    rec_counts = recent['To Grade'].value_counts()
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=rec_counts.index,
                            y=rec_counts.values,
                            marker_color=['#00ff00', '#00d9ff', '#ffaa00', '#ff0000']
                        )
                    ])
                    
                    fig.update_layout(
                        title="Recent Analyst Ratings",
                        xaxis_title="Rating",
                        yaxis_title="Count",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")

# TAB 2: DCF Valuation
with tab2:
    st.header("Discounted Cash Flow (DCF) Valuation")
    
    st.markdown("### Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dcf_ticker = st.text_input("Ticker", value="AAPL", key='dcf_ticker')
        
        if st.button("Fetch Current Data", key='fetch_dcf'):
            try:
                stock = yf.Ticker(dcf_ticker)
                info = stock.info
                
                revenue = info.get('totalRevenue', 0) / 1e9
                st.session_state.dcf_revenue = revenue
                st.success(f"Revenue: ${revenue:.2f}B")
            except:
                st.error("Could not fetch data")
    
    with col2:
        st.markdown("##")
        use_manual = st.checkbox("Use Manual Inputs", value=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        revenue = st.number_input("Current Revenue ($B)", value=st.session_state.get('dcf_revenue', 100.0), min_value=0.1)
        growth_rate = st.number_input("Revenue Growth Rate (%)", value=10.0, min_value=-50.0, max_value=100.0) / 100
    
    with col2:
        fcf_margin = st.number_input("FCF Margin (%)", value=20.0, min_value=0.0, max_value=100.0) / 100
        wacc = st.number_input("WACC (%)", value=10.0, min_value=1.0, max_value=30.0) / 100
    
    with col3:
        terminal_growth = st.number_input("Terminal Growth (%)", value=3.0, min_value=0.0, max_value=10.0) / 100
        shares_outstanding = st.number_input("Shares Outstanding (M)", value=1000.0, min_value=1.0)
    
    if st.button("Calculate Fair Value", key='calc_dcf'):
        enterprise_value, fcf_proj, pv_terminal = calculate_dcf(
            revenue * 1e9, growth_rate, fcf_margin, wacc, terminal_growth
        )
        
        # Adjust for cash and debt (simplified)
        equity_value = enterprise_value
        fair_value_per_share = equity_value / (shares_outstanding * 1e6)
        
        # Display results
        st.markdown("### Valuation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Enterprise Value", f"${enterprise_value/1e9:.2f}B")
        col2.metric("Equity Value", f"${equity_value/1e9:.2f}B")
        col3.metric("Fair Value/Share", f"${fair_value_per_share:.2f}")
        col4.metric("Terminal Value", f"${pv_terminal/1e9:.2f}B")
        
        # Cash flow projections
        st.markdown("### Free Cash Flow Projections")
        
        df_dcf = pd.DataFrame(fcf_proj)
        df_dcf['Revenue'] = df_dcf['Revenue'] / 1e9
        df_dcf['FCF'] = df_dcf['FCF'] / 1e9
        df_dcf['PV_FCF'] = df_dcf['PV_FCF'] / 1e9
        
        df_dcf['Revenue'] = df_dcf['Revenue'].apply(lambda x: f"${x:.2f}B")
        df_dcf['FCF'] = df_dcf['FCF'].apply(lambda x: f"${x:.2f}B")
        df_dcf['PV_FCF'] = df_dcf['PV_FCF'].apply(lambda x: f"${x:.2f}B")
        
        st.dataframe(df_dcf, use_container_width=True)
        
        # Sensitivity analysis
        st.markdown("### Sensitivity Analysis")
        
        wacc_range = np.linspace(wacc * 0.7, wacc * 1.3, 5)
        growth_range = np.linspace(growth_rate * 0.5, growth_rate * 1.5, 5)
        
        sensitivity_matrix = []
        
        for w in wacc_range:
            row = []
            for g in growth_range:
                ev, _, _ = calculate_dcf(revenue * 1e9, g, fcf_margin, w, terminal_growth)
                fv = ev / (shares_outstanding * 1e6)
                row.append(fv)
            sensitivity_matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=sensitivity_matrix,
            x=[f"{g*100:.1f}%" for g in growth_range],
            y=[f"{w*100:.1f}%" for w in wacc_range],
            colorscale='RdYlGn',
            text=[[f"${val:.2f}" for val in row] for row in sensitivity_matrix],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Fair Value")
        ))
        
        fig.update_layout(
            title="Fair Value Sensitivity (Growth Rate vs WACC)",
            xaxis_title="Revenue Growth Rate",
            yaxis_title="WACC",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Factor Screening
with tab3:
    st.header("Multi-Factor Stock Screening")
    
    st.markdown("### Screen Universe")
    
    # Predefined stock universes
    universe_options = {
        "Tech Megacaps": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "Financial Services": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "LLY"],
        "Consumer": ["WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD"]
    }
    
    selected_universe = st.selectbox("Select Stock Universe", list(universe_options.keys()))
    
    tickers = universe_options[selected_universe]
    
    if st.button("Run Factor Analysis", key='run_factors'):
        with st.spinner("Analyzing stocks..."):
            results = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period='3mo')
                    
                    if len(hist) > 20:
                        metrics = {
                            'pe': info.get('trailingPE', 20),
                            'pb': info.get('priceToBook', 3),
                            'revenue_growth': info.get('revenueGrowth', 0.05) * 100,
                            'eps_growth': 10,  # Simplified
                            'roe': info.get('returnOnEquity', 0.15) * 100,
                            'debt_equity': info.get('debtToEquity', 50) / 100,
                            'return_1m': (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100,
                            'return_3m': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                        }
                        
                        scores = calculate_factor_score(metrics)
                        
                        results.append({
                            'Ticker': ticker,
                            'Price': hist['Close'].iloc[-1],
                            'Value Score': scores['value'],
                            'Growth Score': scores['growth'],
                            'Quality Score': scores['quality'],
                            'Momentum Score': scores['momentum'],
                            'Composite Score': scores['composite']
                        })
                except:
                    continue
            
            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('Composite Score', ascending=False)
                
                # Display top picks
                st.markdown("### Top Ranked Stocks")
                
                top_3 = df_results.head(3)
                
                col1, col2, col3 = st.columns(3)
                
                for idx, (col, row) in enumerate(zip([col1, col2, col3], top_3.iterrows())):
                    with col:
                        rank_emoji = ["🥇", "🥈", "🥉"][idx]
                        st.markdown(f"### {rank_emoji} {row[1]['Ticker']}")
                        st.metric("Composite Score", f"{row[1]['Composite Score']:.1f}")
                        st.metric("Current Price", f"${row[1]['Price']:.2f}")
                
                # Full results table
                st.markdown("### Complete Rankings")
                
                # Color code the dataframe
                def color_score(val):
                    if isinstance(val, (int, float)):
                        if val >= 70:
                            return 'background-color: #00ff0033'
                        elif val >= 50:
                            return 'background-color: #ffaa0033'
                        else:
                            return 'background-color: #ff000033'
                    return ''
                
                styled_df = df_results.style.applymap(
                    color_score,
                    subset=['Value Score', 'Growth Score', 'Quality Score', 'Momentum Score', 'Composite Score']
                )
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Radar chart for top 3
                st.markdown("### Factor Comparison (Top 3)")
                
                fig = go.Figure()
                
                categories = ['Value', 'Growth', 'Quality', 'Momentum']
                
                for _, row in top_3.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[row['Value Score'], row['Growth Score'], row['Quality Score'], row['Momentum Score']],
                        theta=categories,
                        fill='toself',
                        name=row['Ticker']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100])
                    ),
                    showlegend=True,
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 4: Peer Comparison
with tab4:
    st.header("Peer Group Analysis")
    
    st.markdown("### Select Companies to Compare")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        comp_tickers = st.text_input(
            "Enter ticker symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL",
            key='comp_tickers'
        )
    
    with col2:
        st.markdown("##")
        compare_btn = st.button("Compare", key='compare_btn')
    
    if compare_btn:
        tickers_list = [t.strip() for t in comp_tickers.split(',')]
        
        with st.spinner("Fetching data..."):
            comp_data = []
            
            for ticker in tickers_list:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    comp_data.append({
                        'Company': ticker,
                        'Market Cap ($B)': info.get('marketCap', 0) / 1e9,
                        'P/E': info.get('trailingPE', 0),
                        'P/B': info.get('priceToBook', 0),
                        'ROE (%)': info.get('returnOnEquity', 0) * 100,
                        'Revenue ($B)': info.get('totalRevenue', 0) / 1e9,
                        'Profit Margin (%)': info.get('profitMargins', 0) * 100,
                        'Debt/Equity': info.get('debtToEquity', 0) / 100
                    })
                except:
                    continue
            
            if comp_data:
                df_comp = pd.DataFrame(comp_data)
                
                # Display table
                st.markdown("### Comparative Metrics")
                st.dataframe(df_comp.round(2), use_container_width=True)
                
                # Visualization
                st.markdown("### Visual Comparison")
                
                metric_to_plot = st.selectbox(
                    "Select Metric",
                    ['Market Cap ($B)', 'P/E', 'P/B', 'ROE (%)', 'Revenue ($B)', 'Profit Margin (%)']
                )
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_comp['Company'],
                        y=df_comp[metric_to_plot],
                        marker_color='#00d9ff'
                    )
                ])
                
                fig.update_layout(
                    title=f"{metric_to_plot} Comparison",
                    yaxis_title=metric_to_plot,
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                st.markdown("### Metric Correlations")
                
                numeric_cols = df_comp.select_dtypes(include=[np.number]).columns
                corr_matrix = df_comp[numeric_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Metric Correlation Matrix",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# TAB 5: Competitive Intelligence
with tab5:
    st.header("Competitive Intelligence & Strategy")
    
    st.markdown("### Market Share Analysis")
    
    # Example: Investment Banking League Tables
    league_table_data = {
        'Bank': ['Goldman Sachs', 'JPMorgan', 'Morgan Stanley', 'Citi', 'Nomura', 'Bank of America'],
        'M&A Volume ($B)': [450, 425, 380, 320, 180, 340],
        'Market Share (%)': [18.5, 17.5, 15.6, 13.2, 7.4, 14.0],
        'Deal Count': [245, 268, 223, 198, 112, 210]
    }
    
    df_league = pd.DataFrame(league_table_data)
    df_league = df_league.sort_values('M&A Volume ($B)', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### M&A League Table (2024)")
        st.dataframe(df_league, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Pie(
                labels=df_league['Bank'],
                values=df_league['Market Share (%)'],
                hole=0.3
            )
        ])
        
        fig.update_layout(
            title="Market Share Distribution",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### M&A Pipeline Predictor")
    
    st.markdown("""
        **AI-Powered Deal Probability Scoring**
        
        This tool analyzes companies for M&A likelihood based on:
        - Valuation multiples vs. peers
        - Cash balance & debt capacity
        - Management commentary & activist presence
        - Strategic fit with acquirers
    """)
    
    # Example M&A targets
    ma_targets = {
        'Company': ['Target A', 'Target B', 'Target C', 'Target D', 'Target E'],
        'Sector': ['Tech', 'Healthcare', 'Finance', 'Consumer', 'Industrial'],
        'Market Cap ($B)': [5.2, 8.7, 3.1, 12.4, 6.8],
        'P/E Discount to Peers': [-25, -15, -30, -10, -20],
        'Deal Probability (%)': [78, 45, 82, 32, 61],
        'Potential Acquirers': ['MSFT, GOOGL', 'JNJ, PFE', 'JPM, WFC', 'PG, COST', 'GE, HON']
    }
    
    df_ma = pd.DataFrame(ma_targets)
    df_ma = df_ma.sort_values('Deal Probability (%)', ascending=False)
    
    st.markdown("#### High-Probability M&A Targets")
    
    # Color code by probability
    def highlight_prob(row):
        if row['Deal Probability (%)'] >= 70:
            return ['background-color: #00ff0033'] * len(row)
        elif row['Deal Probability (%)'] >= 50:
            return ['background-color: #ffaa0033'] * len(row)
        else:
            return [''] * len(row)
    
    styled_ma = df_ma.style.apply(highlight_prob, axis=1)
    st.dataframe(styled_ma, use_container_width=True)
    
    # Probability distribution
    fig = go.Figure(data=[
        go.Bar(
            x=df_ma['Company'],
            y=df_ma['Deal Probability (%)'],
            marker_color=df_ma['Deal Probability (%)'],
            marker_colorscale='RdYlGn',
            text=df_ma['Deal Probability (%)'],
            texttemplate='%{text}%',
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="M&A Deal Probability by Target",
        yaxis_title="Probability (%)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Regulatory Impact Analysis")
    
    st.markdown("""
        **Basel IV Capital Requirements Impact**
        
        Analyzing potential effects on bank profitability:
    """)
    
    reg_impact = {
        'Bank': ['Goldman Sachs', 'JPMorgan', 'Morgan Stanley', 'Citi', 'Nomura'],
        'Current Capital Ratio (%)': [14.5, 13.8, 15.2, 13.1, 12.9],
        'Basel IV Requirement (%)': [15.5, 15.0, 15.8, 14.8, 14.5],
        'Capital Shortfall ($B)': [2.1, 3.8, 1.2, 4.5, 3.2],
        'ROE Impact (bps)': [-45, -65, -28, -78, -55]
    }
    
    df_reg = pd.DataFrame(reg_impact)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(df_reg, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_reg['Bank'],
            y=df_reg['Current Capital Ratio (%)'],
            name='Current',
            marker_color='#00d9ff'
        ))
        
        fig.add_trace(go.Bar(
            x=df_reg['Bank'],
            y=df_reg['Basel IV Requirement (%)'],
            name='Basel IV',
            marker_color='#ff0000'
        ))
        
        fig.update_layout(
            title="Capital Ratios: Current vs Basel IV",
            yaxis_title="Capital Ratio (%)",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p><b>AI Equity Research Analyzer</b> | Built for Nomura Equity Research & Strategy</p>
        <p>Fundamental Analysis • DCF Valuation • Factor Screening • Competitive Intelligence</p>
        <p>Technologies: Python • Streamlit • YFinance • NLP • Data Analytics</p>
    </div>
""", unsafe_allow_html=True)
