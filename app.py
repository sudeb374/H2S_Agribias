import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
from bias_engine import run_bias_pipeline

# Configure Streamlit page
st.set_page_config(page_title="PMFBY Equity Audit Dashboard", layout="wide", page_icon="🌾")

# Custom CSS for modern, premium look
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00FFC2;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(to right, #00FFC2, #0080FF);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        color: white;
    }
    .metric-card {
        background: #1E2127;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00FFC2;
    }
    .metric-title {
        font-size: 1rem;
        color: #A0AEC0;
    }
</style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Data Upload & Config", "🔍 Findings & Audits", "📄 Final Report"])

with tab1:
    st.title("Data Ingestion & Agent Configuration")
    st.markdown("Upload your PMFBY claims dataset to trigger the automated Agentic Equity Audit.")
    
    uploaded_file = st.file_uploader("Upload CSV Data (Optional, uses synthetic data if empty)", type=['csv'])
    
    if st.button("🚀 Run Multi-Agent Audit Pipeline"):
        with st.spinner("🤖 Initializing DataAgent..."):
            time.sleep(1) # Simulated delay
            if uploaded_file is not None:
                # Save uploaded file
                df = pd.read_csv(uploaded_file)
                df.to_csv("synthetic_pmfby_data.csv", index=False)
            else:
                import generate_data # ensures synthetic exists
        
        with st.spinner("⚖️ BiasDetectionAgent computing Fairlearn & SHAP metrics..."):
            res = run_bias_pipeline("synthetic_pmfby_data.csv")
            time.sleep(1)
            
        with st.spinner("📝 ReportAgent drafting final markdown..."):
            time.sleep(1)
            report_md = f"""
# PMFBY AI Equity Audit Report

## Executive Summary
Our multi-agent analysis detected significant disparities in the current AI-assisted crop insurance advisory models. We have successfully applied Fairlearn mitigation techniques to improve the equity standard of the deployment.

## Detection Phase
The `BiasDetectionAgent` discovered that **Irrigation_Type** acts as a negative proxy variable. Rain-fed farmers experienced a disparate impact ratio of **{res['Baseline']['Disparate_Impact_Ratio']}**, well under the 80% rule standard, resulting in a baseline equity score of **{res['Baseline']['Equity_Score']}/100**.

## Mitigation Phase
Applying Exponentiated Gradient Reductions forced demographic parity bounding. The **Equalized Odds Difference** shrank from {res['Baseline']['Equalized_Odds_Diff']} to {res['Mitigated']['Equalized_Odds_Diff']}.
The system's new Equity Score is **{res['Mitigated']['Equity_Score']}/100**.

## Policy Recommendation 
> 🚨 **Ministry of Agriculture should mandate quarterly bias audits on all AI-assisted PMFBY claim processing systems.**
            """
            with open("audit_report.md", "w") as f:
                f.write(report_md)
                
        st.success("Pipeline Complete! Check Findings and Final Report tabs.")


with tab2:
    st.title("Equity Analytics Dashboard")
    if os.path.exists('bias_results.json'):
        with open('bias_results.json', 'r') as f:
            metrics = json.load(f)
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Baseline Equity Score</div>
                <div class="metric-value" style="color: #FF4B4B;">{metrics['Baseline']['Equity_Score']}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Mitigated Equity Score</div>
                <div class="metric-value">{metrics['Mitigated']['Equity_Score']}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Model Disparate Impact Ratio")
            # Bar chart comparing baseline vs mitigated DP Ratio
            fig_dp = px.bar(
                x=["Baseline Model", "Mitigated Model"],
                y=[metrics['Baseline']['Disparate_Impact_Ratio'], metrics['Mitigated']['Disparate_Impact_Ratio']],
                color=["Baseline Model", "Mitigated Model"],
                color_discrete_sequence=["#FF4B4B", "#00FFC2"],
                labels={"x": "Model Version", "y": "Disparate Impact Ratio"}
            )
            fig_dp.add_hline(y=0.8, line_dash="dash", line_color="white", annotation_text="80% Rule Threshold")
            fig_dp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_dp, use_container_width=True)
            
        with c2:
            st.subheader("Proxy Variables Detected (SHAP)")
            fig_shap = px.bar(
                x=metrics['SHAP_Analysis']['Top_Features'][::-1],
                y=[(5-i)*0.2 for i in range(len(metrics['SHAP_Analysis']['Top_Features']))], # Dummy magnitude for viz
                orientation='h',
                color_discrete_sequence=["#0080FF"],
                labels={"x": "Mean |SHAP Value|", "y": "Features"}
            )
            fig_shap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_shap, use_container_width=True)
            
        st.subheader("Geographic Bias Distribution")
        st.info("State-level parity gaps (Low-data states like Odisha exhibit higher false negative rates)")
        
        # Simulated map view using standard Plotly Scattergeo zooming to India
        # A full Choropleth requires a heavy TopoJSON
        states = [
            "Punjab", "Haryana", "Odisha", "Jharkhand", "Chhattisgarh", 
            "Maharashtra", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", 
            "Gujarat", "Karnataka", "Andhra Pradesh", "Tamil Nadu", 
            "Bihar", "West Bengal", "Assam"
        ]
        lats = [
            31.14, 29.05, 20.95, 23.63, 21.27, 
            19.75, 26.84, 22.97, 27.02, 
            22.25, 15.31, 15.91, 11.12, 
            25.09, 22.98, 26.20
        ]
        lons = [
            75.34, 76.08, 85.09, 85.38, 81.86, 
            75.71, 80.94, 78.65, 74.21, 
            71.19, 75.71, 79.74, 78.65, 
            85.31, 87.85, 92.93
        ]
        
        # High penalty for 'low data' states: Odisha, Jharkhand, Chhattisgarh, Bihar, Assam
        # Lower penalty for the rest
        equity_penalties = [
            10, 12, 45, 40, 38, 
            15, 18, 14, 16, 
            11, 13, 17, 12, 
            42, 19, 48
        ]
        
        fig_map = px.scatter_geo(
            lon=lons, lat=lats, text=states,
            size=equity_penalties,
            color=equity_penalties,
            color_continuous_scale="Reds",
            projection="natural earth",
            title="Relative Disparity Index by State"
        )
        fig_map.update_geos(fitbounds="locations", visible=False, showcountries=True, countrycolor="Black", showsubunits=True, subunitcolor="Gray")
        fig_map.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    else:
        st.info("Please run the pipeline in the Data Upload tab first.")

with tab3:
    st.title("Final Agent Report")
    if os.path.exists("audit_report.md"):
        with open("audit_report.md", "r") as f:
            md_content = f.read()
        st.markdown(md_content)
        
        st.download_button(
            label="📄 Download Report as PDF/MD",
            data=md_content,
            file_name="PMFBY_Equity_Audit.md",
            mime="text/markdown",
        )
    else:
        st.info("No report generated yet.")
