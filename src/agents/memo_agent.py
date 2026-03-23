import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

from src.agents.underwriting_agent import (
    compute_financial_metrics,
    compute_noi,
    compute_dscr,
    interpret_dscr
)

from src.rag.retriever import ask_question



# =====================================================
# STRENGTH DETECTOR
# =====================================================

def detect_strengths(metrics, dscr):

    strengths = []

    if dscr and dscr > 1.4:
        strengths.append("strong loan coverage")

    if metrics.get("avg_rent",0) > 5000:
        strengths.append("strong rental income")

    if metrics.get("rent_std",0) < 1500:
        strengths.append("stable rental pricing")

    if metrics.get("avg_lease_years",0) > 5:
        strengths.append("long lease duration")

    if metrics.get("total_expense",999999) < 450000:
        strengths.append("efficient operating costs")

    return strengths



# =====================================================
# RISK DETECTOR
# =====================================================

def detect_risks(metrics, dscr):

    risks = []

    if dscr and dscr < 1.25:
        risks.append("weak loan coverage")

    if metrics.get("rent_std",0) > 2000:
        risks.append("rent volatility risk")

    if metrics.get("total_expense",0) > 500000:
        risks.append("high operating expense burden")

    if metrics.get("avg_lease_years",0) < 4:
        risks.append("short lease duration risk")

    return risks



# =====================================================
# RECOMMENDATION LOGIC
# =====================================================

def generate_recommendation(strengths, risks):

    if len(strengths) > len(risks):
        return "Favorable investment candidate"

    elif len(risks) > len(strengths):
        return "Requires further due diligence"

    else:
        return "Moderate investment profile"



# =====================================================
# CONFIDENCE SCORE
# =====================================================

def confidence_score(metrics):

    score = 0.5

    if "avg_rent" in metrics:
        score += 0.1

    if "total_expense" in metrics:
        score += 0.1

    if "avg_lease_years" in metrics:
        score += 0.1

    return min(score,0.95)



# =====================================================
# MEMO GENERATOR
# =====================================================

def generate_memo(property_id):

    query = f"financial summary {property_id}"

    answer, results = ask_question(query)


    metrics = compute_financial_metrics(results)

    noi = compute_noi(metrics)

    dscr = compute_dscr(
        noi,
        metrics.get("avg_interest_rate",6)
    )


    strengths = detect_strengths(metrics, dscr)

    risks = detect_risks(metrics, dscr)

    recommendation = generate_recommendation(
        strengths,
        risks
    )


    confidence = confidence_score(metrics)



    # ===============================
    # CONTROLLED LLM PROMPT
    # ===============================

    llm_prompt = f"""

You are a senior real estate investment analyst.

Use ONLY the provided financial metrics.

Do NOT recalculate values.

Explain investment quality clearly.


PROPERTY

{property_id}


VERIFIED FINANCIAL METRICS (GROUND TRUTH)

NOI: {noi}

DSCR: {dscr}

Average Monthly Rent: {metrics.get("avg_rent")}

Total Operating Expense: {metrics.get("total_expense")}

Average Lease Years: {metrics.get("avg_lease_years")}


Explain:

investment strength
key financial drivers
risk interpretation

Do not invent numbers.
"""


    from src.rag.retriever import load_llm

    llm = load_llm()

    explanation = llm.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[

            {

                "role":"user",

                "content":llm_prompt

            }

        ]

    ).choices[0].message.content



    memo = f"""

INVESTMENT MEMO

Property:
{property_id}


EXECUTIVE SUMMARY

{explanation}


KEY METRICS

NOI:
{noi}

DSCR:
{dscr}

Loan Risk:
{interpret_dscr(dscr)}


Average Rent:
{metrics.get("avg_rent")}

Total Expense:
{metrics.get("total_expense")}

Lease Duration:
{metrics.get("avg_lease_years")}



STRENGTHS

{chr(10).join(["• " + s for s in strengths])}



RISKS

{chr(10).join(["• " + r for r in risks])}



RECOMMENDATION

{recommendation}



CONFIDENCE SCORE

{round(confidence,2)}

"""


    return memo


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    memo = generate_memo("PROP003")

    print(memo)