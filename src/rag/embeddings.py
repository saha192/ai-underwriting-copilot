import os
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from chunking import process_documents



# =====================================================
# CONFIG
# =====================================================

EMBED_MODEL = "all-MiniLM-L6-v2"

VECTOR_STORE_DIR = "vector_store"

INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "faiss.index"
)

METADATA_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "chunks.pkl"
)



# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

def load_model():

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBED_MODEL
    )

    return model



# =====================================================
# DOMAIN-AWARE ENRICHMENT
# =====================================================

def enrich_for_embedding(chunk):

    md = chunk.metadata

    enrichment = ""


    # -------------------------------------
    # PROPERTY CONTEXT
    # -------------------------------------

    enrichment += f"""
    real estate property {md.get('property_id')}
    investment asset building financial analysis
    """


    # -------------------------------------
    # FINANCIAL CONTEXT
    # -------------------------------------

    if md.get("contains_financial_data"):

        enrichment += """
        financial metric revenue expense cost price valuation accounting income cashflow profit loss operating expense capital expense
        """


    # -------------------------------------
    # LEASE CONTEXT
    # -------------------------------------

    if md.get("contains_lease_info"):

        enrichment += """
        lease tenant rental agreement occupancy duration sqft property leasing contract expiry rent escalation
        """


    # -------------------------------------
    # LOAN CONTEXT
    # -------------------------------------

    if md.get("contains_loan_terms"):

        enrichment += """
        loan interest rate debt financing amortization dscr credit risk lending principal payment coverage ratio underwriting
        """


    # -------------------------------------
    # RISK CONTEXT
    # -------------------------------------

    if md.get("contains_risk_info"):

        enrichment += """
        investment risk volatility uncertainty downside exposure sensitivity scenario analysis financial risk macroeconomic risk
        """


    # -------------------------------------
    # DOCUMENT STRUCTURE SIGNALS
    # -------------------------------------

    enrichment += f"""
    document type {md.get('document_type')}
    chunk type {md.get('chunk_type')}
    """


    # -------------------------------------
    # NUMERIC SIGNAL BOOST
    # -------------------------------------

    text_lower = chunk.text.lower()

    if any(keyword in text_lower for keyword in ["cost", "expense", "amount"]):

        enrichment += " numeric financial value monetary amount price cost expense "

    if "rent" in text_lower:

        enrichment += " rental income monthly rent revenue lease payment "

    if "interest" in text_lower:

        enrichment += " borrowing rate financing cost interest percentage debt rate "


    return enrichment + "\n" + chunk.text



# =====================================================
# EMBEDDING PIPELINE
# =====================================================

def embed_chunks(chunks, model):

    print("\nGenerating enriched embeddings...")

    texts = [

        enrich_for_embedding(c)

        for c in chunks

    ]


    embeddings = model.encode(

        texts,

        show_progress_bar=True

    )


    return np.array(embeddings)



# =====================================================
# BUILD FAISS INDEX
# =====================================================

def build_index(embeddings):

    print("\nBuilding FAISS index...")

    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(

        dimension

    )


    index.add(

        embeddings

    )


    return index



# =====================================================
# SAVE VECTOR STORE
# =====================================================

def save_vector_store(index, chunks):

    print("\nSaving vector store...")


    os.makedirs(

        VECTOR_STORE_DIR,

        exist_ok=True

    )


    # save FAISS index

    faiss.write_index(

        index,

        INDEX_FILE

    )


    # save chunk metadata

    metadata = [

        {

            "text": c.text,

            "metadata": c.metadata,

            "chunk_id": c.chunk_id

        }

        for c in chunks

    ]


    with open(

        METADATA_FILE,

        "wb"

    ) as f:

        pickle.dump(

            metadata,

            f

        )


    print(f"\nSaved index → {INDEX_FILE}")

    print(f"Saved metadata → {METADATA_FILE}")



# =====================================================
# MAIN PIPELINE
# =====================================================

def create_vector_store(data_path):

    print("\nProcessing documents...")

    chunks = process_documents(

        data_path

    )


    print(f"\nTotal chunks: {len(chunks)}")


    model = load_model()


    embeddings = embed_chunks(

        chunks,

        model

    )


    index = build_index(

        embeddings

    )


    save_vector_store(

        index,

        chunks

    )


    print("\nVector store ready!")



# =====================================================
# RUN SCRIPT
# =====================================================

if __name__ == "__main__":

    data_path = r"D:\real_estate_rag\data\raw"

    create_vector_store(

        data_path

    )