import sys
import os
import streamlit as st
import pandas as pd
import re


# ========================================
# PATH FIX
# ========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)


# ========================================
# IMPORT BACKEND
# ========================================

from src.rag.retriever import ask_question

from src.agents.underwriting_agent import (
    compute_financial_metrics,
    compute_noi,
    compute_dscr,
    interpret_dscr
)

from src.agents.memo_agent import generate_memo

from src.agents.portfolio_agent import rank_portfolio


# ========================================
# PAGE CONFIG
# ========================================

st.set_page_config(
    page_title="AI Underwriting Copilot",
    layout="wide"
)


# ========================================
# MODERN UI STYLE
# ========================================

st.markdown("""

<style>

body {
background-color:#0b1120;
color:white;
font-family: Inter, sans-serif;
}

.metric-card {
background:#020617;
padding:18px;
border-radius:12px;
border:1px solid #1e293b;
text-align:center;
}

.metric-value {
font-size:26px;
font-weight:600;
}

.metric-label {
font-size:13px;
color:#94a3b8;
}

.section-card {
background:#020617;
padding:22px;
border-radius:14px;
border:1px solid #1e293b;
line-height:1.6;
}

hr {
border:none;
border-top:1px solid #1e293b;
margin-top:25px;
margin-bottom:25px;
}

</style>

""", unsafe_allow_html=True)



# ========================================
# HEADER
# ========================================

st.title("AI Real Estate Underwriting Copilot")

st.caption(
"Decision intelligence for real estate investments powered by AI"
)


# ========================================
# NAVIGATION
# ========================================

tab1, tab2, tab3 = st.tabs([
    "Copilot",
    "Investment Memo",
    "Portfolio Analytics"
])


# ========================================
# HELPERS
# ========================================

def extract_metrics(results):

    metrics = compute_financial_metrics(results)

    noi = compute_noi(metrics)

    dscr = compute_dscr(
        noi,
        metrics.get("avg_interest_rate",6)
    )

    return metrics, noi, dscr



def risk_indicator(dscr):

    if dscr is None:
        return "⚪ Unknown"

    if dscr > 1.5:
        return "🟢 Strong"

    elif dscr > 1.25:
        return "🟡 Moderate"

    else:
        return "🔴 Risky"



def clean_summary(text):

    text = text.replace("INVESTMENT MEMO","")
    text = text.replace("EXECUTIVE SUMMARY","")

    text = text.split("KEY METRICS")[0]

    text = re.sub(r"\n{2,}", "\n", text)

    text = text.strip()

    replacements = {

        "Based on the provided financial metrics": "",

        "here's an assessment of the property's investment quality": "",

        "Considering these factors": "",

        "relatively": "",

        "suggests that": "indicates"

    }

    for k,v in replacements.items():

        text = text.replace(k,v)

    return text.strip()



def generate_driver_insights(metrics, dscr):

    insights = []

    if dscr and dscr > 1.5:
        insights.append("Strong debt servicing capacity")

    if metrics.get("avg_rent",0) > 4500:
        insights.append("Stable rental income")

    if metrics.get("avg_lease_years",0) > 5:
        insights.append("Long lease duration supports predictable cash flow")

    if metrics.get("total_expense",0) < 500000:
        insights.append("Operating expenses within efficient range")

    if metrics.get("tenant_count",0) >= 20:
        insights.append("Healthy tenant diversification")

    return insights



# ========================================
# COPILOT
# ========================================

with tab1:

    st.subheader("Ask investment question")

    query = st.text_input(
        "Example: Compare PROP001 and PROP003"
    )


    if st.button("Analyze"):

        with st.spinner("Running analysis..."):

            answer, results = ask_question(query)

            metrics, noi, dscr = extract_metrics(results)


        summary = clean_summary(answer)


        # -----------------------------
        # SUMMARY
        # -----------------------------

        st.markdown("### Investment Summary")

        st.markdown(
            f"""
<div class='section-card'>

{summary}

</div>
""",
            unsafe_allow_html=True
        )


        # -----------------------------
        # METRICS
        # -----------------------------

        st.markdown("### Key Metrics")

        col1, col2, col3, col4 = st.columns(4)


        col1.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{round(noi) if noi else "NA"}</div>
<div class='metric-label'>NOI</div>
</div>
""", unsafe_allow_html=True)


        col2.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{dscr if dscr else "NA"}</div>
<div class='metric-label'>DSCR</div>
</div>
""", unsafe_allow_html=True)


        col3.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{risk_indicator(dscr)}</div>
<div class='metric-label'>Risk Level</div>
</div>
""", unsafe_allow_html=True)


        col4.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{round(metrics.get("avg_rent",0))}</div>
<div class='metric-label'>Avg Rent</div>
</div>
""", unsafe_allow_html=True)


        # -----------------------------
        # INSIGHTS
        # -----------------------------

        insights = generate_driver_insights(metrics, dscr)

        if insights:

            st.markdown("### Key Drivers")

            st.markdown(
                "<div class='section-card'>",
                unsafe_allow_html=True
            )

            for i in insights:
                st.write(f"• {i}")

            st.markdown("</div>", unsafe_allow_html=True)



        # -----------------------------
        # EVIDENCE
        # -----------------------------

        with st.expander("Evidence"):

            for r in results:

                st.markdown(
                    f"**{r['metadata']['document_type']} ({r['metadata']['property_id']})**"
                )

                st.code(r["text"])

                st.markdown("---")



# ========================================
# MEMO
# ========================================

with tab2:

    st.subheader("Generate investment memo")

    property_id = st.selectbox(
        "Select property",
        [
            "PROP001",
            "PROP002",
            "PROP003",
            "PROP004"
        ]
    )


    if st.button("Generate"):

        memo = generate_memo(property_id)

        summary = clean_summary(memo)


        st.markdown("### Executive Summary")

        st.markdown(
            f"""
<div class='section-card'>

{summary}

</div>
""",
            unsafe_allow_html=True
        )



# ========================================
# PORTFOLIO
# ========================================

with tab3:

    st.subheader("Portfolio ranking")


    if st.button("Run portfolio analysis"):

        ranked = rank_portfolio()

        df = pd.DataFrame(ranked)


        st.dataframe(df, use_container_width=True)


        best = ranked[0]


        st.markdown("### Top Investment")


        col1, col2, col3 = st.columns(3)


        col1.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{best['property']}</div>
<div class='metric-label'>Property</div>
</div>
""", unsafe_allow_html=True)


        col2.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{best['dscr']}</div>
<div class='metric-label'>DSCR</div>
</div>
""", unsafe_allow_html=True)


        col3.markdown(f"""
<div class='metric-card'>
<div class='metric-value'>{best['score']}</div>
<div class='metric-label'>Score</div>
</div>
""", unsafe_allow_html=True)



# ========================================
# FOOTER
# ========================================

st.markdown("---")

st.caption(
"AI Underwriting Copilot • Real Estate Decision Intelligence"
)