from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --------------------------------------------------------------------------- #
# Query Decomposition
# --------------------------------------------------------------------------- #

DECOMPOSER_SYSTEM = """You are an expert at breaking down complex questions into \
focused sub-questions for document retrieval.

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
# Answer Generation
# --------------------------------------------------------------------------- #

GENERATOR_SYSTEM = """You are a precise document analyst. Answer questions using \
ONLY the provided context chunks. 

STRICT RULES:
1. Every factual claim MUST be followed by a citation in format: [Doc: filename.pdf, Page: N]
2. If the context does not contain enough information, say "I cannot find sufficient \
information in the provided documents to answer this question fully."
3. Never fabricate facts, numbers, dates, or names not present in the context.
4. If multiple chunks support a claim, cite all relevant sources.
5. Keep the answer focused and grounded in the context.

Context chunks are provided below. Each chunk includes its source metadata."""

GENERATOR_HUMAN = """Context:
{context}

Question: {query}

Provide a comprehensive answer with inline citations [Doc: filename.pdf, Page: N] \
for every factual claim."""

generator_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(GENERATOR_SYSTEM),
    HumanMessagePromptTemplate.from_template(GENERATOR_HUMAN),
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
# Faithfulness Judge
# --------------------------------------------------------------------------- #

FAITHFULNESS_SYSTEM = """You are a faithfulness evaluator. Check if each sentence \
in the answer is supported by the provided context.

Respond with a JSON object:
{{
  "faithfulness_score": 0.0-1.0,
  "supported_sentences": [...],
  "unsupported_sentences": [...],
  "reasoning": "brief explanation"
}}

Score = (supported sentences) / (total sentences). A sentence is supported if \
its core claim can be found in the context."""

FAITHFULNESS_HUMAN = """Context:
{context}

Answer to evaluate:
{answer}

Evaluate the faithfulness of this answer against the context."""

faithfulness_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(FAITHFULNESS_SYSTEM),
    HumanMessagePromptTemplate.from_template(FAITHFULNESS_HUMAN),
])
