import os
import uuid
import json
import pandas as pd
import numpy as np
import datetime

from typing import Dict
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader



# =====================================================
# DATA STRUCTURE
# =====================================================

@dataclass
class Chunk:

    text: str
    metadata: Dict
    chunk_id: str



# =====================================================
# JSON SERIALIZATION FIX
# =====================================================

def make_json_serializable(obj):

    if isinstance(obj, dict):

        return {

            k: make_json_serializable(v)

            for k, v in obj.items()

        }

    elif isinstance(obj, list):

        return [

            make_json_serializable(v)

            for v in obj

        ]

    elif isinstance(obj, pd.Timestamp):

        return obj.strftime("%Y-%m-%d")

    elif isinstance(obj, datetime.date):

        return obj.strftime("%Y-%m-%d")

    elif isinstance(obj, np.integer):

        return int(obj)

    elif isinstance(obj, np.floating):

        return float(obj)

    elif pd.isna(obj):

        return None

    else:

        return obj



# =====================================================
# DOCUMENT TYPE IDENTIFICATION
# =====================================================

def identify_document_type(filename):

    name = filename.lower()

    if "rent_roll" in name:
        return "rent_roll"

    elif "operating" in name:
        return "operating_statement"

    elif "construction" in name:
        return "construction_budget"

    elif "loan_terms" in name:
        return "loan_terms"

    elif "property_description" in name:
        return "property_description"

    else:
        return "unknown"



# =====================================================
# PROPERTY ID DETECTION
# =====================================================

def extract_property_id_from_filename(filename):

    possible_ids = [

        "PROP001",
        "PROP002",
        "PROP003",
        "PROP004"

    ]

    for prop in possible_ids:

        if prop in filename:

            return prop

    return "PORTFOLIO_LEVEL"



# =====================================================
# SMART SEMANTIC TAGGING
# =====================================================

def tag_chunk_semantics(text):

    text_lower = text.lower()

    tags = {

        "contains_financial_data": False,
        "contains_lease_info": False,
        "contains_loan_terms": False,
        "contains_risk_info": False

    }


    financial_keywords = [

        "rent",
        "expense",
        "cost",
        "income",
        "noi",
        "debt",
        "interest",
        "budget",
        "amount"
    ]


    lease_keywords = [

        "lease",
        "tenant",
        "sqft"
    ]


    loan_keywords = [

        "interest rate",
        "loan amount",
        "amortization",
        "debt service",
        "dscr"
    ]


    risk_keywords = [

        "risk",
        "volatility",
        "exposure",
        "sensitivity",
        "competition"
    ]


    if any(k in text_lower for k in financial_keywords):
        tags["contains_financial_data"] = True


    if any(k in text_lower for k in lease_keywords):
        tags["contains_lease_info"] = True


    if any(k in text_lower for k in loan_keywords):
        tags["contains_loan_terms"] = True


    if any(k in text_lower for k in risk_keywords):
        tags["contains_risk_info"] = True


    return tags



# =====================================================
# TABLE CHUNKING
# =====================================================

def chunk_dataframe(df, document_type, source_file):

    chunks = []

    grouped = df.groupby("property_id")

    ROWS_PER_CHUNK = 4


    for property_id, group in grouped:

        rows = group.to_dict(orient="records")


        for i in range(0, len(rows), ROWS_PER_CHUNK):

            subset = rows[i:i + ROWS_PER_CHUNK]


            text = f"""
DOCUMENT_TYPE: {document_type}

PROPERTY_ID: {property_id}

SOURCE: financial_table

CONTENT:
"""


            for row in subset:

                clean_row = make_json_serializable(row)

                row_json = json.dumps(

                    clean_row,

                    ensure_ascii=False

                )

                text += row_json + "\n"


            tags = tag_chunk_semantics(text)


            metadata = {

                "property_id": property_id,

                "document_type": document_type,

                "chunk_type": "table",

                "source_file": source_file,

                **tags

            }


            chunks.append(

                Chunk(

                    text=text,

                    metadata=metadata,

                    chunk_id=str(uuid.uuid4())

                )

            )


    return chunks



# =====================================================
# DERIVED TOTALS
# =====================================================

def create_financial_summary(df, document_type, source_file):

    summaries = []

    grouped = df.groupby("property_id")


    for property_id, group in grouped:

        numeric_cols = group.select_dtypes(

            include="number"

        )


        totals = numeric_cols.sum().to_dict()


        text = f"""
DOCUMENT_TYPE: derived_totals_{document_type}

PROPERTY_ID: {property_id}

SOURCE: aggregated_financial_summary

CONTENT:
"""


        for k, v in totals.items():

            text += f"{k}_total = {v}\n"


        text += f"row_count = {len(group)}\n"


        tags = tag_chunk_semantics(text)


        summaries.append(

            Chunk(

                text=text,

                metadata={

                    "property_id": property_id,

                    "document_type": f"derived_totals_{document_type}",

                    "chunk_type": "derived_financial",

                    "source_file": source_file,

                    **tags

                },

                chunk_id=str(uuid.uuid4())

            )

        )


    return summaries



# =====================================================
# RENT ROLL METRICS
# =====================================================

def generate_rent_roll_metrics(df):

    metrics = []

    grouped = df.groupby("property_id")


    for property_id, group in grouped:

        avg_rent = group["monthly_rent_usd"].mean()

        max_rent = group["monthly_rent_usd"].max()

        min_rent = group["monthly_rent_usd"].min()

        tenant_count = len(group)


        lease_duration = (

            pd.to_datetime(group["lease_end"])

            -

            pd.to_datetime(group["lease_start"])

        ).dt.days / 365


        avg_lease_years = lease_duration.mean()


        text = f"""
DOCUMENT_TYPE: derived_metrics_rent_roll

PROPERTY_ID: {property_id}

SOURCE: financial_analysis

CONTENT:

avg_monthly_rent = {round(avg_rent,2)}

max_rent = {max_rent}

min_rent = {min_rent}

tenant_count = {tenant_count}

avg_lease_years = {round(avg_lease_years,2)}
"""


        tags = tag_chunk_semantics(text)


        metrics.append(

            Chunk(

                text=text,

                metadata={

                    "property_id": property_id,

                    "document_type": "derived_metrics_rent_roll",

                    "chunk_type": "derived_financial",

                    **tags

                },

                chunk_id=str(uuid.uuid4())

            )

        )


    return metrics



# =====================================================
# OPERATING METRICS
# =====================================================

def generate_operating_metrics(df):

    metrics = []

    grouped = df.groupby("property_id")


    for property_id, group in grouped:

        total_expense = group["annual_amount_usd"].sum()


        max_category = group.loc[

            group["annual_amount_usd"].idxmax()

        ]


        text = f"""
DOCUMENT_TYPE: derived_metrics_operating_statement

PROPERTY_ID: {property_id}

SOURCE: financial_analysis

CONTENT:

total_operating_expense = {total_expense}

highest_cost_category = {max_category['category']}

highest_cost_value = {max_category['annual_amount_usd']}
"""


        tags = tag_chunk_semantics(text)


        metrics.append(

            Chunk(

                text=text,

                metadata={

                    "property_id": property_id,

                    "document_type": "derived_metrics_operating_statement",

                    "chunk_type": "derived_financial",

                    **tags

                },

                chunk_id=str(uuid.uuid4())

            )

        )


    return metrics



# =====================================================
# TEXT CHUNKING
# =====================================================

def chunk_text(text, property_id, document_type, source_file):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=60

    )


    splits = splitter.split_text(text)


    chunks = []


    for split in splits:


        formatted_text = f"""
DOCUMENT_TYPE: {document_type}

PROPERTY_ID: {property_id}

SOURCE: narrative_text

CONTENT:

{split}
"""


        tags = tag_chunk_semantics(formatted_text)


        chunks.append(

            Chunk(

                text=formatted_text,

                metadata={

                    "property_id": property_id,

                    "document_type": document_type,

                    "chunk_type": "text",

                    "source_file": source_file,

                    **tags

                },

                chunk_id=str(uuid.uuid4())

            )

        )


    return chunks



# =====================================================
# PDF LOADER
# =====================================================

def load_pdf(path):

    reader = PdfReader(path)

    text = ""


    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted + "\n"


    return text



# =====================================================
# MAIN PIPELINE
# =====================================================

def process_documents(raw_data_path):

    all_chunks = []


    for file in os.listdir(raw_data_path):

        path = os.path.join(raw_data_path, file)

        doc_type = identify_document_type(file)

        property_id = extract_property_id_from_filename(file)


        print(f"Processing: {file}")


        if file.endswith(".xlsx"):


            df = pd.read_excel(path)


            table_chunks = chunk_dataframe(

                df,

                doc_type,

                file

            )


            totals_chunks = create_financial_summary(

                df,

                doc_type,

                file

            )


            all_chunks.extend(table_chunks)

            all_chunks.extend(totals_chunks)


            if doc_type == "rent_roll":

                all_chunks.extend(

                    generate_rent_roll_metrics(df)

                )


            if doc_type == "operating_statement":

                all_chunks.extend(

                    generate_operating_metrics(df)

                )



        elif file.endswith(".txt"):


            with open(path, "r", encoding="utf-8") as f:

                text = f.read()


            all_chunks.extend(

                chunk_text(

                    text,

                    property_id,

                    doc_type,

                    file

                )

            )



        elif file.endswith(".pdf"):


            text = load_pdf(path)


            all_chunks.extend(

                chunk_text(

                    text,

                    property_id,

                    doc_type,

                    file

                )

            )



    print(f"\nTotal chunks created: {len(all_chunks)}")


    return all_chunks



# =====================================================
# TEST RUN
# =====================================================

if __name__ == "__main__":


    raw_data_path = r"D:\real_estate_rag\data\raw"


    chunks = process_documents(

        raw_data_path

    )


    print("\n--------------------")


    print("\nSAMPLE CHUNK:\n")


    print(chunks[0].text)


    print("\nMETADATA:\n")


    print(chunks[0].metadata)


    print("\nCHUNK ID:\n")


    print(chunks[0].chunk_id)