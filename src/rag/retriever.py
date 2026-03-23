import os
import re
import pickle
import numpy as np
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq



# =====================================================
# INIT
# =====================================================

load_dotenv()


EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_FILE = "vector_store/faiss.index"

METADATA_FILE = "vector_store/chunks.pkl"

TOP_K = 6


GROQ_MODEL = "llama-3.1-8b-instant"



# =====================================================
# GLOBAL CACHE (IMPORTANT OPTIMIZATION)
# =====================================================

_index = None
_metadata = None
_embed_model = None
_llm = None



# =====================================================
# LOAD COMPONENTS (CACHED)
# =====================================================

def load_vector_store():

    global _index, _metadata


    if _index is None:

        print("Loading FAISS index...")

        _index = faiss.read_index(INDEX_FILE)


        with open(METADATA_FILE, "rb") as f:

            _metadata = pickle.load(f)


    return _index, _metadata



def load_embedding_model():

    global _embed_model


    if _embed_model is None:

        print("Loading embedding model...")

        _embed_model = SentenceTransformer(

            EMBED_MODEL

        )


    return _embed_model



def load_llm():

    global _llm


    if _llm is None:

        _llm = Groq(

            api_key=os.getenv("GROQ_API_KEY")

        )


    return _llm



# =====================================================
# QUERY UNDERSTANDING
# =====================================================

def classify_query(query):

    q = query.lower()


    if any(x in q for x in [

        "average",

        "mean",

        "total",

        "sum",

        "calculate",

        "compute"

    ]):

        return "calculation"


    elif any(x in q for x in [

        "risk",

        "safe",

        "good investment",

        "viable"

    ]):

        return "risk_analysis"


    elif any(x in q for x in [

        "compare",

        "difference",

        "better"

    ]):

        return "comparison"


    else:

        return "lookup"



# =====================================================
# PROPERTY DETECTION
# =====================================================

def extract_property_ids(query):

    matches = re.findall(

        r"PROP\d{3}",

        query.upper()

    )


    return matches



# =====================================================
# QUERY REWRITING
# =====================================================

def rewrite_query(query, intent):

    base = query


    if intent == "risk_analysis":

        base += """

        Evaluate financial risk using:

        rent stability

        operating expense

        loan terms

        lease duration

        """


    elif intent == "comparison":

        base += """

        compare financial strength

        compare expense ratio

        compare rent stability

        compare loan burden

        """


    elif intent == "calculation":

        base += """

        identify financial values

        compute derived metrics

        perform financial ratios

        """


    return base



# =====================================================
# EMBEDDING
# =====================================================

def embed_query(query, model):

    return model.encode([query])



# =====================================================
# SEARCH
# =====================================================

def semantic_search(query_vector, index, metadata):

    distances, indices = index.search(

        np.array(query_vector),

        TOP_K * 4

    )


    results = []


    for dist, idx in zip(

        distances[0],

        indices[0]

    ):

        chunk = metadata[idx]


        results.append(

            {

                "text": chunk["text"],

                "metadata": chunk["metadata"],

                "score": float(dist)

            }

        )


    return results



# =====================================================
# FILTER BY PROPERTY
# =====================================================

def filter_by_property(results, property_ids):

    if not property_ids:

        return results


    filtered = []


    for r in results:

        if r["metadata"]["property_id"] in property_ids:

            filtered.append(r)


    return filtered



# =====================================================
# BOOST IMPORTANT CHUNKS
# =====================================================

def boost_scores(results):

    for r in results:

        md = r["metadata"]


        if md["chunk_type"] == "derived_financial":

            r["score"] *= 0.80


        if md.get("contains_risk_info"):

            r["score"] *= 0.90


        if md.get("contains_loan_terms"):

            r["score"] *= 0.90


    return results



# =====================================================
# FINAL RANKING
# =====================================================

def rank_results(results):

    ranked = sorted(

        results,

        key=lambda x: x["score"]

    )


    return ranked[:TOP_K]



# =====================================================
# STRUCTURED CONTEXT BUILDER
# =====================================================

def build_structured_context(results):

    sections = {

        "financial": [],

        "loan": [],

        "risk": [],

        "property": []

    }


    for r in results:

        md = r["metadata"]


        if md.get("contains_loan_terms"):

            sections["loan"].append(r["text"])


        elif md.get("contains_risk_info"):

            sections["risk"].append(r["text"])


        elif md.get("contains_financial_data"):

            sections["financial"].append(r["text"])


        else:

            sections["property"].append(r["text"])



    context = ""


    if sections["financial"]:

        context += "\nFINANCIAL DATA\n"

        context += "\n".join(sections["financial"])


    if sections["loan"]:

        context += "\n\nLOAN TERMS\n"

        context += "\n".join(sections["loan"])


    if sections["risk"]:

        context += "\n\nRISK FACTORS\n"

        context += "\n".join(sections["risk"])


    if sections["property"]:

        context += "\n\nPROPERTY DETAILS\n"

        context += "\n".join(sections["property"])


    return context



# =====================================================
# PROMPT BUILDER
# =====================================================

def build_prompt(query, context, intent):

    instructions = """

You are an expert real estate financial analyst.

Follow these rules:

use numbers from context

perform calculations when needed

compare properties objectively

identify financial risks

avoid hallucinating values

explain reasoning clearly

"""

    if intent == "risk_analysis":

        instructions += """

Focus on:

expense volatility

lease expiry risk

loan burden

"""


    if intent == "comparison":

        instructions += """

Compare investment strength logically.

"""


    return f"""

{instructions}

CONTEXT:

{context}


QUESTION:

{query}

Provide structured financial reasoning.
"""



# =====================================================
# LLM CALL
# =====================================================

def ask_llm(prompt, client):

    completion = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {

                "role": "user",

                "content": prompt

            }

        ]

    )


    return completion.choices[0].message.content



# =====================================================
# MAIN PIPELINE
# =====================================================

def ask_question(query):

    index, metadata = load_vector_store()

    embed_model = load_embedding_model()

    llm = load_llm()


    intent = classify_query(query)


    property_ids = extract_property_ids(query)


    rewritten_query = rewrite_query(

        query,

        intent

    )


    query_vector = embed_query(

        rewritten_query,

        embed_model

    )


    results = semantic_search(

        query_vector,

        index,

        metadata

    )


    results = filter_by_property(

        results,

        property_ids

    )


    results = boost_scores(results)


    results = rank_results(results)


    context = build_structured_context(

        results

    )


    prompt = build_prompt(

        query,

        context,

        intent

    )


    answer = ask_llm(

        prompt,

        llm

    )


    return answer, results



# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    query = "Compare PROP001 and PROP003 investment strength"


    answer, results = ask_question(query)


    print("\nANSWER:\n")

    print(answer)


    print("\nDOCUMENT TYPES:\n")


    for r in results:

        print(

            r["metadata"]["document_type"]

        )