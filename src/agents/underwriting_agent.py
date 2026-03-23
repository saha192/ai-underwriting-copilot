import sys
import os
import re
import json
import numpy as np

# ensure imports work
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

from src.rag.retriever import ask_question



# =====================================================
# PROPERTY DETECTION
# =====================================================

def extract_property_ids(query):

    return re.findall(
        r"PROP\d{3}",
        query.upper()
    )



# =====================================================
# JSON PARSER
# =====================================================

def parse_json_lines(text):

    records = []

    for line in text.split("\n"):

        line = line.strip()

        if line.startswith("{") and line.endswith("}"):

            try:
                records.append(json.loads(line))
            except:
                pass

    return records



# =====================================================
# KEY VALUE PARSER
# =====================================================

def parse_key_value_lines(text):

    pairs = {}

    matches = re.findall(
        r"([a-zA-Z_]+)\s*=\s*([\d\.]+)",
        text
    )

    for k,v in matches:

        try:
            pairs[k.lower()] = float(v)
        except:
            pass

    return pairs



# =====================================================
# HYBRID EXTRACTION
# =====================================================

def extract_structured_financials(results):

    rents = []
    expenses = []
    interest_rates = []

    derived = {}

    for r in results:

        text = r["text"]


        # JSON rows
        records = parse_json_lines(text)

        for rec in records:

            for k,v in rec.items():

                key = k.lower()

                if not isinstance(v,(int,float)):
                    continue


                if "rent" in key:
                    rents.append(v)


                elif "expense" in key or "cost" in key:
                    expenses.append(v)


                elif "interest" in key:
                    interest_rates.append(v)



        # derived metrics
        kv_pairs = parse_key_value_lines(text)

        derived.update(kv_pairs)



    return rents, expenses, interest_rates, derived



# =====================================================
# FINANCIAL METRICS
# =====================================================

def compute_financial_metrics(results):

    rents, expenses, interest_rates, derived = extract_structured_financials(results)


    metrics = {}


    # rent
    if "avg_monthly_rent" in derived:

        metrics["avg_rent"] = derived["avg_monthly_rent"]

    elif rents:

        metrics["avg_rent"] = round(
            float(np.mean(rents)),2
        )


    # tenants
    if "tenant_count" in derived:

        metrics["tenant_count"] = derived["tenant_count"]


    # lease duration
    if "avg_lease_years" in derived:

        metrics["avg_lease_years"] = derived["avg_lease_years"]


    # expense
    if "total_operating_expense" in derived:

        metrics["total_expense"] = derived["total_operating_expense"]

    elif expenses:

        metrics["total_expense"] = round(
            float(np.sum(expenses)),2
        )


    # rent variability
    if rents:

        metrics["rent_std"] = round(
            float(np.std(rents)),2
        )


    # interest
    if interest_rates:

        metrics["avg_interest_rate"] = round(
            float(np.mean(interest_rates)),2
        )


    return metrics



# =====================================================
# NOI CALCULATOR
# =====================================================

def compute_noi(metrics):

    rent = metrics.get("avg_rent")
    tenants = metrics.get("tenant_count")
    expense = metrics.get("total_expense")

    if not rent or not tenants or not expense:

        return None


    annual_revenue = rent * tenants * 12

    noi = annual_revenue - expense

    return round(noi,2)



# =====================================================
# DSCR CALCULATOR
# =====================================================

def compute_dscr(

    noi,
    interest_rate,
    loan_amount=5_000_000
):

    if not noi or not interest_rate:

        return None


    annual_debt_service = loan_amount * (interest_rate/100)

    dscr = noi / annual_debt_service

    return round(dscr,3)



# =====================================================
# DSCR INTERPRETATION
# =====================================================

def interpret_dscr(dscr):

    if dscr is None:
        return "Unknown"


    if dscr < 1:

        return "High Risk"


    elif dscr < 1.25:

        return "Moderate Risk"


    elif dscr < 1.5:

        return "Acceptable"


    else:

        return "Strong"



# =====================================================
# RISK SCORE
# =====================================================

def compute_risk_score(metrics, dscr):

    score = 0


    expense = metrics.get("total_expense")

    rent_std = metrics.get("rent_std",0)


    if expense:

        if expense > 500000:
            score += 1


    if rent_std > 1800:

        score += 1


    if dscr:

        if dscr < 1.2:
            score += 2

        elif dscr < 1.4:
            score += 1


    return score



# =====================================================
# MEMO FORMATTER
# =====================================================

def format_memo(

    query,
    answer,
    metrics,
    noi,
    dscr,
    risk_score
):

    if risk_score >= 3:

        risk_level = "High"

    elif risk_score == 2:

        risk_level = "Medium"

    else:

        risk_level = "Low"



    memo = f"""

INVESTMENT MEMO


Query:

{query}



KEY METRICS

Average Rent:
{metrics.get("avg_rent","NA")} USD


Tenant Count:
{metrics.get("tenant_count","NA")}


Total Expense:
{metrics.get("total_expense","NA")} USD


NOI:
{noi}


DSCR:
{dscr}


Loan Risk:
{interpret_dscr(dscr)}



OVERALL RISK LEVEL

{risk_level}



AI ANALYSIS

{answer}


"""


    return memo



# =====================================================
# CRITIC
# =====================================================

def critique(metrics):

    issues = []

    if "avg_rent" not in metrics:

        issues.append("rent data missing")

    if "total_expense" not in metrics:

        issues.append("expense data missing")

    return issues



# =====================================================
# MAIN AGENT
# =====================================================

def run_agent(query):

    print("\nPLANNING\n")

    properties = extract_property_ids(query)

    print("properties detected:", properties)


    print("\nRETRIEVING\n")

    answer, results = ask_question(query)


    print("\nCOMPUTING METRICS\n")

    metrics = compute_financial_metrics(results)

    print(metrics)


    noi = compute_noi(metrics)

    dscr = compute_dscr(
        noi,
        metrics.get("avg_interest_rate",6)
    )


    risk_score = compute_risk_score(
        metrics,
        dscr
    )


    print("\nNOI:", noi)

    print("DSCR:", dscr)

    print("Loan Risk:", interpret_dscr(dscr))


    critique_notes = critique(metrics)

    print("\nCRITIQUE\n")

    print(critique_notes)


    memo = format_memo(

        query,
        answer,
        metrics,
        noi,
        dscr,
        risk_score
    )


    return memo



# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    query = "Compare PROP001 and PROP003 investment strength"

    result = run_agent(query)

    print("\nFINAL OUTPUT\n")

    print(result)