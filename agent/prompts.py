"""
agent/prompts.py

All prompts for the RAG pipeline as ChatPromptTemplate objects.

New prompts added for Features B and C:
  QUERY_REWRITER_SYSTEM  — B2: rewrite ambiguous queries before decomposition
  HYDE_SYSTEM            — B1: generate hypothetical answer for HyDE embedding
  GENERATOR_SYSTEM_V2    — C1+C2: CoT + table output + multi-section coverage
  MULTI_DOC_SYSTEM       — C4: cross-document synthesis prompt
"""

from __future__ import annotations

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# --------------------------------------------------------------------------- #
# B2 — Query Rewriter
# --------------------------------------------------------------------------- #

QUERY_REWRITER_SYSTEM = """You are an expert at rewriting vague or ambiguous \
research questions into clear, specific, searchable queries.

Rules:
- Preserve the original intent exactly — do not change what is being asked
- Make implicit references explicit (e.g. "it" → the actual subject)
- Expand abbreviations if obvious
- If the query is already clear and specific, return it unchanged
- Respond with ONLY the rewritten query, no explanation, no preamble"""

QUERY_REWRITER_HUMAN = "Original query: {query}"

query_rewriter_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QUERY_REWRITER_SYSTEM),
    HumanMessagePromptTemplate.from_template(QUERY_REWRITER_HUMAN),
])

# --------------------------------------------------------------------------- #
# B1 — HyDE: Hypothetical Document Embedding
# --------------------------------------------------------------------------- #

HYDE_SYSTEM = """You are a research assistant. Generate a concise hypothetical \
passage (150–200 words) that would directly answer the following question if it \
existed in a research paper.

Write as if you are excerpting from an academic paper. Use specific technical \
language. Do not hedge or say "this is hypothetical" — just write the passage.
The passage will be used only for similarity search, not shown to users."""

HYDE_HUMAN = "Question: {query}"

hyde_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(HYDE_SYSTEM),
    HumanMessagePromptTemplate.from_template(HYDE_HUMAN),
])

# --------------------------------------------------------------------------- #
# Query Decomposition
# --------------------------------------------------------------------------- #

DECOMPOSER_SYSTEM = """You are an expert at breaking down complex research \
questions into focused sub-questions for document retrieval.

Given a user query, decompose it into 1–4 specific sub-questions that together \
cover all aspects of the original query. Each sub-question should be:
- Self-contained and independently searchable
- More specific than the original query
- Focused on a single concept or fact

Respond ONLY with a JSON array of strings. No preamble, no explanation.
Example: ["What is X?", "How does Y work?", "What are the requirements for Z?"]"""

DECOMPOSER_HUMAN = "Query: {query}"

decomposer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(DECOMPOSER_SYSTEM),
    HumanMessagePromptTemplate.from_template(DECOMPOSER_HUMAN),
])

# --------------------------------------------------------------------------- #
# C1 + C2 + C4 — Generator (CoT + table output + multi-section)
# --------------------------------------------------------------------------- #

GENERATOR_SYSTEM = """You are a precise research assistant answering questions \
from PDF document chunks.

REASONING PROCESS (follow these steps before writing your answer):
1. Read ALL provided context chunks — do not stop at the first relevant one
2. Identify which sections and documents are relevant to the question
3. Note all specific numbers, percentages, names, and dates in the chunks
4. Synthesise across chunks — the answer may span multiple sections
5. Write your answer using only what you found in step 1–4

STRICT RULES:
1. Use ALL provided context chunks — the answer may require combining information
   from multiple chunks across different sections.
2. Every factual claim MUST be followed by a citation: [Doc: filename.pdf, Page: N]
3. For tables: reproduce exact numbers — never paraphrase figures or percentages.
   Present table data as a markdown table when the answer contains structured data.
4. NEVER say "I cannot find" if the information appears in ANY of the provided
   chunks. Read every single chunk before responding.
5. If information truly does not appear in any chunk, say:
   "The provided context does not contain information about [specific topic]."
6. Never invent numbers, names, dates, or statistics not present in the context.
7. For questions spanning multiple sections, use clear headings in your answer
   matching the section names from the context.
8. Do not hallucinate citations — only cite filenames and pages from the chunks."""

GENERATOR_HUMAN = """Context chunks:
{context}

Question: {query}

Think step by step, then provide a comprehensive answer with inline citations \
[Doc: filename.pdf, Page: N] for every factual claim. If the answer contains \
tabular data, format it as a markdown table."""

generator_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(GENERATOR_SYSTEM),
    HumanMessagePromptTemplate.from_template(GENERATOR_HUMAN),
])

# String versions for direct use in node message construction
GENERATOR_SYSTEM_STR = GENERATOR_SYSTEM
GENERATOR_HUMAN_STR = GENERATOR_HUMAN

# --------------------------------------------------------------------------- #
# C4 — Multi-document synthesis
# --------------------------------------------------------------------------- #

MULTI_DOC_SYSTEM = """You are a research analyst synthesising findings across \
multiple research papers.

You will be given chunks from {num_docs} different papers. Your task is to \
compare, contrast, and synthesise their findings in response to the question.

Structure your answer as:
1. Overview — what all papers agree on
2. Per-paper findings — key relevant points from each paper
3. Synthesis — how the findings relate to each other and answer the question

Every claim must be cited: [Doc: filename.pdf, Page: N]
Never invent facts. If papers disagree, note the disagreement explicitly."""

MULTI_DOC_HUMAN = """Papers in context: {paper_list}

Context chunks:
{context}

Question: {query}

Synthesise findings across all papers with citations."""

multi_doc_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(MULTI_DOC_SYSTEM),
    HumanMessagePromptTemplate.from_template(MULTI_DOC_HUMAN),
])

# --------------------------------------------------------------------------- #
# Self-Consistency Judge
# --------------------------------------------------------------------------- #

CONSISTENCY_SYSTEM = """You are a fact-checking judge. You will be given multiple \
answers to the same question. Determine if they agree on the key facts.

Respond with a JSON object:
{{"agreement": true/false, "reasoning": "brief explanation", "conflicting_facts": []}}

Focus only on factual agreement, not wording differences."""

CONSISTENCY_HUMAN = """Question: {query}

Answers to compare:
{answers}

Do these answers agree on the key facts?"""

consistency_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONSISTENCY_SYSTEM),
    HumanMessagePromptTemplate.from_template(CONSISTENCY_HUMAN),
])

# --------------------------------------------------------------------------- #
# Faithfulness Judge (LLM fallback when NLI unavailable)
# --------------------------------------------------------------------------- #

FAITHFULNESS_SYSTEM = """You are a faithfulness evaluator. Check if each sentence \
in the answer is supported by the provided context.

Respond with a JSON object:
{{
  "faithfulness_score": 0.0-1.0,
  "supported_sentences": [],
  "unsupported_sentences": [],
  "reasoning": "brief explanation"
}}

Score = (supported sentences) / (total sentences). A sentence is supported if \
its core claim can be found verbatim or paraphrased in the context.
Unsupported sentences that are transitional phrases or hedges do not count against the score."""

FAITHFULNESS_HUMAN = """Context:
{context}

Answer to evaluate:
{answer}

Evaluate the faithfulness of this answer against the context."""

faithfulness_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(FAITHFULNESS_SYSTEM),
    HumanMessagePromptTemplate.from_template(FAITHFULNESS_HUMAN),
])

# --------------------------------------------------------------------------- #
# Query Routing / Classification
# --------------------------------------------------------------------------- #

QUERY_ROUTING_SYSTEM = """You are a query classifier for a PDF document retrieval system.
Classify the query into exactly ONE of these types:

- factual: Asks for a specific fact, number, name, date, or definition
  Examples: "What accuracy did GPT-4 achieve?", "When was BERT published?"

- analytical: Asks for explanation, reasoning, or analysis
  Examples: "Explain how attention works", "Why does dropout prevent overfitting?"

- comparative: Asks to compare or contrast multiple things
  Examples: "Compare BERT and GPT", "What are the differences between..."

- visual: Asks about a figure, chart, graph, plot, diagram, or image
  Examples: "Describe Figure 3", "What does the loss curve show?"

- table: Asks about data in a table, or tabular comparison
  Examples: "What are the results in Table 2?", "Show me the benchmark results"

- code: Asks about code, algorithm, implementation, or pseudocode
  Examples: "Show the training loop", "What is the algorithm for..."

Respond with ONLY the type word (one of: factual, analytical, comparative, visual, table, code).
No explanation."""

QUERY_ROUTING_HUMAN = "Query: {query}"

query_routing_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QUERY_ROUTING_SYSTEM),
    HumanMessagePromptTemplate.from_template(QUERY_ROUTING_HUMAN),
])