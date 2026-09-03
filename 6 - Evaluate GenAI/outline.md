```
START — evaluate whether the TechMart GenAI chatbot is ready to ship
  │
  ▼
[B0] Setup
  ├── load OpenAI credentials
  └── initialize the OpenAI client
  │
  ▼
[B1] Create the evaluation example
  ├── user asks about the electronics return policy
  ├── expected answer contains the correct policy
  ├── gpt-5-nano generates a chatbot answer
  └── create a second answer that sounds convincing
        but invents price matching and a lifetime warranty
  │
  ▼
[B1] Why traditional metrics fail
  ├── valid answers may use completely different wording
  ├── word overlap does not guarantee factual correctness
  └── a mostly correct answer can still contain one dangerous hallucination
  │
  ▼
[B2] Evaluate in layers
  │
  ├── code-based checks
  │     ├── response length
  │     ├── banned words and competitor mentions
  │     ├── system-prompt leakage
  │     └── output-format validation
  │
  │     Fast and deterministic, but unable to judge meaning.
  │
  └── LLM-as-a-Judge
        ├── give the judge the question, response and rubric
        ├── score accuracy, helpfulness, tone and completeness
        └── return {"score": 1–5, "reason": "..."}
  │
  ▼
[B3] G-Eval with DeepEval
  ├── represent each example as an LLMTestCase
  ├── define custom evaluation criteria or explicit evaluation steps
  ├── measure empathy and factual accuracy
  └── convert scores into pass/fail using thresholds
        │
        └── an answer may pass empathy while failing accuracy
            → no single metric describes overall quality
  │
  ▼
[B4] Expand to seven synthetic RAG test cases
  └── each case contains:
        { question,
          retrieved context,
          expected answer,
          actual answer,
          known scenario }
  │
  ▼
[B4] Run the DeepEval test suite
  ├── Answer Relevancy — does the response answer the question?
  ├── Faithfulness — does it contradict the context?
  ├── Contextual Relevancy — did retrieval find useful information?
  ├── Groundedness — is every claim supported by the context?
  └── Professional Tone — is the response appropriate?
  │
  ▼
[B4] Important metric trap
  ├── invented price matching:
  │     Faithfulness = 1.00
  │     Groundedness = 0.00
  └── the invented claim contradicts nothing,
        but it is also supported by nothing

        → "not contradicted" does not mean "grounded"
  │
  ▼
[B5] Diagnose the RAG pipeline with RAGAS
  │
  ├── RETRIEVER METRICS
  │     ├── Context Precision — how much retrieved content is useful?
  │     └── Context Recall — were all required facts retrieved?
  │
  └── GENERATOR METRICS
        ├── Faithfulness — are answer claims supported by context?
        └── Answer Relevancy — does the answer address the question?
  │
  ▼
[B5] Locate the source of failure
  ├── poor precision/recall → retrieval problem
  ├── poor faithfulness → hallucinating generator
  └── poor answer relevancy → generation/prompt problem

  Note: DeepEval and RAGAS use different definitions of "faithfulness".
  Always inspect the metric definition, not just its name.
  │
  ▼
════════════════════ production evaluation ════════════════════
  │
  ▼
[B6] Instrument the chatbot with Langfuse
  │
  ├── retrieve_context(query)
  ├── generate_response(query, context)
  └── techmart_chatbot(query)
        │
        └── @observe() records retrieval, model calls,
            inputs, outputs, timings and metadata
  │
  ▼
[B6] Score production traces
  ├── an LLM judge scores response helpfulness
  ├── attach the score and explanation to the Langfuse trace
  └── inspect quality, latency and behavior in the dashboard
  │
  ▼
[B6] Production feedback loop
  └── bad production trace
        → add it to the Langfuse golden dataset
          → run it through DeepEval
            → fix the chatbot
              → deploy and monitor again
  │
  ▼
[B7] Red-team the chatbot
  ├── manually test prompt injection, prompt leakage,
  │   bias, unsafe requests and data extraction
  └── use DeepTeam to generate attacks and measure
        Bias · Misinformation · PII Leakage · Prompt Leakage
  │
  ▼
[B8] Final release gate
  ├── run multiple quality and safety metrics
  ├── calculate the overall pass rate
  └── decide:
        ≥90%   → deploy with monitoring
        70–89% → fix failures first
        <70%   → do not deploy
  │
  ▼
END RESULT
A complete GenAI evaluation lifecycle:

development tests
  → RAG diagnosis
    → safety testing
      → production tracing
        → production failures become new regression tests
```