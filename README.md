# AI Real Estate Underwriting Copilot

AI-powered decision intelligence system for real estate investment analysis using **LLMs, Retrieval-Augmented Generation (RAG), and Agentic AI workflows**.

This project demonstrates how modern AI systems can automate underwriting workflows including **DSCR analysis, financial document understanding, investment memo generation, and portfolio ranking**.

---

## Live Demo

Streamlit Application:

https://ai-real-estate-underwriting-agent.streamlit.app/

Try queries like:

Compare PROP001 and PROP003  
Which property has highest DSCR?  
Generate investment memo for PROP002  
Which property is highest risk?  

---

## Problem Statement

Real estate underwriting requires analyzing large volumes of financial and operational documents:

rent rolls  
operating statements  
loan terms  
construction budgets  
property descriptions  

Manual analysis is:

time-consuming  
error-prone  
difficult to scale  

This project demonstrates how AI can automate financial reasoning and support faster investment decisions.

---

## Key Features

### 1. AI Document Intelligence

Processes structured and unstructured real estate financial documents:

Excel financial statements  
PDF loan documents  
property descriptions  
construction budgets  

Extracts relevant financial signals for downstream reasoning.

---

### 2. Retrieval-Augmented Generation (RAG)

Implements a production-style RAG pipeline:

semantic chunking optimized for financial data  
vector embeddings using Sentence Transformers  
FAISS vector database for fast similarity search  
metadata-aware filtering by property ID  
structured context generation for grounded LLM responses  

Ensures answers are supported by retrieved evidence.

---

### 3. AI Underwriting Agent

Agent-based architecture performs financial reasoning:

Net Operating Income (NOI) calculation  
Debt Service Coverage Ratio (DSCR) computation  
expense analysis  
lease duration interpretation  
risk assessment  

Combines symbolic financial logic with LLM reasoning.

---

### 4. Investment Memo Generator

Automatically produces structured investment summaries:

executive summary  
key financial metrics  
risk interpretation  
investment recommendation  

Output resembles internal credit memos used in real estate investment firms.

---

### 5. Portfolio Ranking Engine

Ranks multiple properties based on investment strength using:

DSCR strength  
expense efficiency  
rent stability  
lease duration  
tenant diversification  

Produces decision-ready ranking output.

---

### 6. Streamlit Copilot Interface

Interactive AI interface for:

natural language investment queries  
memo generation  
portfolio analytics  
evidence inspection  

Designed to resemble production-grade analytics software.

---

## System Architecture
# AI Real Estate Underwriting Copilot

AI-powered decision intelligence system for real estate investment analysis using **LLMs, Retrieval-Augmented Generation (RAG), and Agentic AI workflows**.

This project demonstrates how modern AI systems can automate underwriting workflows including **DSCR analysis, financial document understanding, investment memo generation, and portfolio ranking**.

---

## Live Demo

Streamlit Application:

https://ai-real-estate-underwriting-agent.streamlit.app/

Try queries like:

Compare PROP001 and PROP003  
Which property has highest DSCR?  
Generate investment memo for PROP002  
Which property is highest risk?  

---

## Problem Statement

Real estate underwriting requires analyzing large volumes of financial and operational documents:

rent rolls  
operating statements  
loan terms  
construction budgets  
property descriptions  

Manual analysis is:

time-consuming  
error-prone  
difficult to scale  

This project demonstrates how AI can automate financial reasoning and support faster investment decisions.

---

## Key Features

### 1. AI Document Intelligence

Processes structured and unstructured real estate financial documents:

Excel financial statements  
PDF loan documents  
property descriptions  
construction budgets  

Extracts relevant financial signals for downstream reasoning.

---

### 2. Retrieval-Augmented Generation (RAG)

Implements a production-style RAG pipeline:

semantic chunking optimized for financial data  
vector embeddings using Sentence Transformers  
FAISS vector database for fast similarity search  
metadata-aware filtering by property ID  
structured context generation for grounded LLM responses  

Ensures answers are supported by retrieved evidence.

---

### 3. AI Underwriting Agent

Agent-based architecture performs financial reasoning:

Net Operating Income (NOI) calculation  
Debt Service Coverage Ratio (DSCR) computation  
expense analysis  
lease duration interpretation  
risk assessment  

Combines symbolic financial logic with LLM reasoning.

---

### 4. Investment Memo Generator

Automatically produces structured investment summaries:

executive summary  
key financial metrics  
risk interpretation  
investment recommendation  

Output resembles internal credit memos used in real estate investment firms.

---

### 5. Portfolio Ranking Engine

Ranks multiple properties based on investment strength using:

DSCR strength  
expense efficiency  
rent stability  
lease duration  
tenant diversification  

Produces decision-ready ranking output.

---

### 6. Streamlit Copilot Interface

Interactive AI interface for:

natural language investment queries  
memo generation  
portfolio analytics  
evidence inspection  

Designed to resemble production-grade analytics software.

---

## System Architecture
Document Processing Layer
PDF / Excel ingestion
financial table parsing
semantic chunking

Embedding Layer
Sentence Transformers embeddings

Vector Database
FAISS similarity search

RAG Pipeline
query rewriting
metadata filtering
structured context generation

Agent Layer
financial reasoning tools
DSCR calculator
risk scoring logic
memo generation agent

Application Layer
Streamlit interface
decision dashboard
portfolio analytics

Document Processing Layer
PDF / Excel ingestion
financial table parsing
semantic chunking

Embedding Layer
Sentence Transformers embeddings

Vector Database
FAISS similarity search

RAG Pipeline
query rewriting
metadata filtering
structured context generation

Agent Layer
financial reasoning tools
DSCR calculator
risk scoring logic
memo generation agent

Application Layer
Streamlit interface
decision dashboard
portfolio analytics
