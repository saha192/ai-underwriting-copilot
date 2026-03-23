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
    compute_dscr
)

from src.rag.retriever import ask_question



# ==========================
# PROPERTY LIST
# ==========================

PROPERTIES = [

    "PROP001",
    "PROP002",
    "PROP003",
    "PROP004"

]



# ==========================
# NORMALIZATION
# ==========================

def normalize(values):

    v = np.array(values)

    if np.max(v) == np.min(v):

        return np.ones_like(v)

    return (v - np.min(v)) / (np.max(v) - np.min(v))



# ==========================
# SCORE CALCULATION
# ==========================

def compute_property_score(metrics, dscr):

    score = 0

    if dscr:

        score += 0.35 * min(dscr / 2, 1)


    rent = metrics.get("avg_rent",0)

    score += 0.20 * min(rent / 6000, 1)


    expense = metrics.get("total_expense",0)

    score += 0.20 * (1 - min(expense / 600000,1))


    tenants = metrics.get("tenant_count",0)

    score += 0.10 * min(tenants / 30,1)


    lease = metrics.get("avg_lease_years",0)

    score += 0.15 * min(lease / 6,1)


    return round(score,3)



# ==========================
# MAIN PIPELINE
# ==========================

def rank_portfolio():

    results = []


    for prop in PROPERTIES:

        print("\nAnalyzing", prop)


        query = f"financial summary {prop}"


        answer, retrieved = ask_question(query)


        metrics = compute_financial_metrics(retrieved)


        noi = compute_noi(metrics)


        dscr = compute_dscr(
            noi,
            metrics.get("avg_interest_rate",6)
        )


        score = compute_property_score(metrics, dscr)


        results.append({

            "property": prop,
            "score": score,
            "dscr": dscr,
            "noi": noi,
            "rent": metrics.get("avg_rent")

        })



    ranked = sorted(

        results,

        key=lambda x: x["score"],

        reverse=True

    )


    return ranked



# ==========================
# FORMAT OUTPUT
# ==========================

def print_ranking(ranked):

    print("\nPORTFOLIO RANKING\n")


    for i,r in enumerate(ranked,1):

        print(

            i,
            r["property"],
            "score:", r["score"],
            "DSCR:", r["dscr"]
        )



# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    ranked = rank_portfolio()

    print_ranking(ranked)