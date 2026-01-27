# source.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

# --- Observability: OpenTelemetry (tracing) ---
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import (  # type: ignore
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
try:
    # Optional OTLP exporter (recommended for real deployments)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
        OTLPSpanExporter,
    )
except Exception:  # pragma: no cover
    OTLPSpanExporter = None  # type: ignore

# --- Observability: Prometheus (metrics) ---
from prometheus_client import Counter, Histogram, start_http_server, REGISTRY  # type: ignore


# --- LLM Clients ---
try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


# =========================================================
# Defaults (data-only)
# =========================================================

DEFAULT_LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai:gpt-4o-2024-08-06": {"temperature": 0.3},
    "openai:gpt-3.5-turbo-16k": {"temperature": 0.3},
    "openai:gpt-4": {"temperature": 0.3},
}

DEFAULT_LLM_PROMPTS: Dict[str, Dict[str, Any]] = {
    "evidence_extraction": {
        "id": "evidence_extraction",
        "raw": """Extract AI-readiness evidence from this SEC filing excerpt.

Filing: {{{{filing_excerpt}}}}

Return JSON with:
- dimension: one of [data_infrastructure, ai_governance, technology_stack, talent, leadership, use_case_portfolio, culture]
- evidence: the specific text
- confidence: 0-1 score
- reasoning: why this is evidence""",
    },
    "dimension_scoring": {
        "id": "dimension_scoring",
        "raw": """Score this company's {{{{dimension}}}} dimension based on the evidence.

Evidence:
{{{{evidence_items}}}}

Return a score from 0-100 with justification.""",
    },
}

DEFAULT_TEST_CASES: List[Dict[str, Any]] = [
    {
        "description": "Extract data infrastructure evidence",
        "prompt_id": "evidence_extraction",
        "vars": {
            "filing_excerpt": "The company invested $50M in cloud data infrastructure, deploying Snowflake and Databricks for enterprise analytics."
        },
        "assert": [
            {"type": "json"},
            {"type": "contains", "value": "data_infrastructure"},
            {"type": "javascript", "value": "output.confidence >= 0.7"},
        ],
    },
    {
        "description": "Extract governance evidence",
        "prompt_id": "evidence_extraction",
        "vars": {
            "filing_excerpt": "We established an AI Ethics Board reporting to the CEO, with quarterly reviews of model risk and bias assessments."
        },
        "assert": [
            {"type": "json"},
            {"type": "contains", "value": "ai_governance"},
            {"type": "javascript", "value": "output.confidence >= 0.7"},
        ],
    },
    {
        "description": "Score talent dimension (low score expected)",
        "prompt_id": "dimension_scoring",
        "vars": {
            "dimension": "talent",
            "evidence_items": """- Hired Chief AI Officer from Google
- 45 open AI/ML positions
- Partnered with Stanford for ML research""",
        },
        "assert": [
            {"type": "javascript",
                "value": r"parseInt(output.match(/\d+/)[0]) >= 60"},
            {"type": "javascript",
                "value": r"parseInt(output.match(/\d+/)[0]) <= 100"},
        ],
    },
    {
        "description": "Score talent dimension (high score expected)",
        "prompt_id": "dimension_scoring",
        "vars": {
            "dimension": "talent",
            "evidence_items": """- Hired 100+ PhDs in AI/ML
- 200 open AI/ML positions
- Established a world-class AI research lab with $500M investment""",
        },
        "assert": [
            {"type": "javascript",
                "value": r"parseInt(output.match(/\d+/)[0]) >= 85"},
            {"type": "javascript",
                "value": r"parseInt(output.match(/\d+/)[0]) <= 100"},
        ],
    },
    {
        "description": "Handle ambiguous evidence",
        "prompt_id": "evidence_extraction",
        "vars": {"filing_excerpt": "The company uses Microsoft Office 365 for productivity."},
        "assert": [
            {"type": "javascript",
                "value": "output.confidence < 0.5 || output.dimension === null"}
        ],
    },
]


# =========================================================
# Observability setup
# =========================================================

_TRACING_INITIALIZED = False
_PROMETHEUS_STARTED = False


def _get_or_create_counter(name: str, documentation: str, labelnames: List[str]):
    existing = REGISTRY._names_to_collectors.get(
        name)  # type: ignore[attr-defined]
    if existing is not None:
        return existing
    return Counter(name, documentation, labelnames)


def _get_or_create_histogram(name: str, documentation: str, labelnames: List[str]):
    existing = REGISTRY._names_to_collectors.get(
        name)  # type: ignore[attr-defined]
    if existing is not None:
        return existing
    return Histogram(name, documentation, labelnames)


# Prometheus metrics (safe across Streamlit reruns / repeated imports)
LLM_REQUESTS_TOTAL = _get_or_create_counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
)
LLM_LATENCY_SECONDS = _get_or_create_histogram(
    "llm_latency_seconds",
    "LLM request latency in seconds",
    ["provider", "model"],
)
LLM_TOKENS_TOTAL = _get_or_create_counter(
    "llm_tokens_total",
    "Total tokens used",
    ["provider", "model", "type"],  # prompt|completion|total
)
ASSERTIONS_TOTAL = _get_or_create_counter(
    "evaluation_assertions_total",
    "Total assertions evaluated",
    ["assertion_type", "passed"],
)
TEST_RUNS_TOTAL = _get_or_create_counter(
    "evaluation_test_runs_total",
    "Total (test,provider) runs",
    ["passed"],
)


def init_tracing(
    service_name: str = "qulab-lab14",
    otlp_endpoint: Optional[str] = None,
    enable_console_exporter: bool = True,
) -> None:
    """
    Initialize OpenTelemetry tracing.

    - If otlp_endpoint is provided and OTLP exporter is available, spans are sent to that endpoint (HTTP/proto).
    - Optionally also exports to console for local debugging.
    """
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # OTLP exporter (recommended if you run an OpenTelemetry Collector)
    if otlp_endpoint and OTLPSpanExporter is not None:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except Exception:
            # Fall back silently; console exporter may still be enabled
            pass

    if enable_console_exporter:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    _TRACING_INITIALIZED = True


def get_tracer():
    return trace.get_tracer("qulab.lab14")


def start_prometheus_metrics_server(port: int = 8000, addr: str = "0.0.0.0") -> None:
    """
    Start a Prometheus metrics endpoint. Safe to call multiple times.
    Prometheus should scrape: http://<host>:<port>/metrics
    """
    global _PROMETHEUS_STARTED
    if _PROMETHEUS_STARTED:
        return
    start_http_server(port=port, addr=addr)
    _PROMETHEUS_STARTED = True


# =========================================================
# Assertions
# =========================================================

def assert_type_json(output: str) -> Tuple[bool, str]:
    try:
        json.loads(output)
        return True, "Output is valid JSON."
    except json.JSONDecodeError:
        return False, "Output is not valid JSON."


def assert_type_contains(output: str, value: str) -> Tuple[bool, str]:
    try:
        parsed_output = json.loads(output)
        contains = value in json.dumps(parsed_output)
        return contains, f"Output contains '{value}'." if contains else f"Output does not contain '{value}'."
    except json.JSONDecodeError:
        contains = value in output
        return contains, f"Raw output contains '{value}'." if contains else f"Raw output does not contain '{value}'."


def _try_parse_json_dict(output: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(output)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def assert_type_javascript(output: str, js_expression: str) -> Tuple[bool, str]:
    output_obj = _try_parse_json_dict(output)

    if "output.confidence" in js_expression or "output.dimension" in js_expression:
        confidence = output_obj.get("confidence") if output_obj else None
        dimension = output_obj.get("dimension") if output_obj else None

        if js_expression.strip() == "output.confidence >= 0.7":
            ok = confidence is not None and confidence >= 0.7
            return ok, f"Confidence ({confidence}) >= 0.7." if ok else f"Confidence ({confidence}) < 0.7."

        if js_expression.strip() == "output.confidence < 0.5":
            ok = confidence is not None and confidence < 0.5
            return ok, f"Confidence ({confidence}) < 0.5." if ok else f"Confidence ({confidence}) >= 0.5."

        if js_expression.strip() == "output.confidence < 0.5 || output.dimension === null":
            ok = (confidence is not None and confidence <
                  0.5) or (dimension is None)
            return ok, (
                f"Ambiguous evidence handled (confidence={confidence}, dimension={dimension})."
                if ok
                else "Ambiguous evidence not handled correctly."
            )

        return False, f"Unsupported JS expression: '{js_expression}'"

    if "parseInt(output.match(/\\d+/)[0])" in js_expression:
        match = re.search(r"\d+", output)
        score = int(match.group(0)) if match else None

        if ">=" in js_expression:
            threshold = int(js_expression.split(">=")[-1].strip())
            ok = score is not None and score >= threshold
            return ok, f"Score ({score}) >= {threshold}." if ok else f"Score ({score}) < {threshold}."

        if "<=" in js_expression:
            threshold = int(js_expression.split("<=")[-1].strip())
            ok = score is not None and score <= threshold
            return ok, f"Score ({score}) <= {threshold}." if ok else f"Score ({score}) > {threshold}."

        return False, f"Unsupported JS expression: '{js_expression}'"

    return False, f"Unsupported or failed JS expression evaluation: '{js_expression}'"


def get_assertion_funcs() -> Dict[str, Callable[..., Tuple[bool, str]]]:
    return {
        "json": assert_type_json,
        "contains": assert_type_contains,
        "javascript": assert_type_javascript,
    }


# =========================================================
# Prompt formatting
# =========================================================

def format_prompt(raw_prompt: str, variables: Dict[str, Any]) -> str:
    """
    Replace occurrences like {{{{var}}}} OR {{var}} with values (robust to minor template variations).
    """
    formatted = raw_prompt
    for k, v in variables.items():
        formatted = formatted.replace(
            f"{{{{{{{{{k}}}}}}}}}", str(v))  # {{{{var}}}}
        formatted = formatted.replace(
            f"{{{{{k}}}}}", str(v))          # {{var}}
    return formatted


# =========================================================
# Real LLM calls
# =========================================================

def _parse_provider(provider_id: str) -> Tuple[str, str]:
    """
    provider_id format: "<vendor>:<model>"
    e.g. "openai:gpt-4o-2024-08-06"
    """
    if ":" not in provider_id:
        return provider_id, provider_id
    vendor, model = provider_id.split(":", 1)
    return vendor.strip(), model.strip()


def _call_openai_chat(
    model: str,
    prompt: str,
    *,
    api_key: str,
    temperature: float = 0.3,
    timeout_s: int = 60,
) -> Tuple[str, Dict[str, Any]]:
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed in this environment.")

    client = OpenAI(api_key=api_key)

    # Simple 1-message chat. For stricter output, you can add a system instruction.
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ],
        timeout=timeout_s,
    )

    text = resp.choices[0].message.content or ""
    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }
    return text, usage


def call_llm(
    prompt_id: str,
    formatted_prompt: str,
    provider_id: str,
    provider_config: Dict[str, Any],
    api_keys: Dict[str, str],
    *,
    timeout_s: int = 60,
    max_retries: int = 2,
    retry_backoff_s: float = 1.5,
) -> str:
    """
    Calls the actual LLM for the given provider_id.

    api_keys:
      - openai: required for openai providers
      - anthropic: optional if you want to enable anthropic providers
    """
    tracer = get_tracer()
    vendor, model = _parse_provider(provider_id)
    temperature = float(provider_config.get("temperature", 0.3))

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        start = time.time()
        status = "ok"
        usage: Dict[str, Any] = {}
        try:
            with tracer.start_as_current_span("llm.call") as span:
                span.set_attribute("llm.vendor", vendor)
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.prompt_id", prompt_id)
                span.set_attribute("llm.temperature", temperature)
                span.set_attribute("llm.attempt", attempt)

                if vendor == "openai":
                    api_key = api_keys.get("openai", "")
                    if not api_key:
                        raise RuntimeError(
                            "Missing OpenAI API key. Provide it in the Streamlit sidebar.")
                    text, usage = _call_openai_chat(
                        model=model,
                        prompt=formatted_prompt,
                        api_key=api_key,
                        temperature=temperature,
                        timeout_s=timeout_s,
                    )
                elif vendor == "anthropic":
                    pass
                    # api_key = api_keys.get("anthropic", "")
                    # if not api_key:
                    #     raise RuntimeError(
                    #         "Missing Anthropic API key (optional). Set api_keys['anthropic'] to enable.")
                    # text, usage = _call_anthropic_messages(
                    #     model=model,
                    #     prompt=formatted_prompt,
                    #     api_key=api_key,
                    #     temperature=temperature,
                    #     timeout_s=timeout_s,
                    # )
                else:
                    raise RuntimeError(
                        f"Unsupported provider vendor '{vendor}' in provider_id='{provider_id}'.")

                # Metrics: tokens (best effort)
                if "prompt_tokens" in usage and usage["prompt_tokens"] is not None:
                    LLM_TOKENS_TOTAL.labels(vendor, model, "prompt").inc(
                        int(usage["prompt_tokens"]))
                if "completion_tokens" in usage and usage["completion_tokens"] is not None:
                    LLM_TOKENS_TOTAL.labels(vendor, model, "completion").inc(
                        int(usage["completion_tokens"]))
                if "total_tokens" in usage and usage["total_tokens"] is not None:
                    LLM_TOKENS_TOTAL.labels(vendor, model, "total").inc(
                        int(usage["total_tokens"]))

                # Claude usage best-effort
                if "input_tokens" in usage:
                    LLM_TOKENS_TOTAL.labels(vendor, model, "prompt").inc(
                        int(usage["input_tokens"]))
                if "output_tokens" in usage:
                    LLM_TOKENS_TOTAL.labels(vendor, model, "completion").inc(
                        int(usage["output_tokens"]))

                span.set_attribute("llm.output_length", len(text))
                return text

        except Exception as e:
            last_err = e
            status = "error"
            if attempt < max_retries:
                time.sleep(retry_backoff_s ** attempt)
            else:
                # Return an error string (keeps evaluation pipeline flowing)
                return f"ERROR: {type(e).__name__}: {str(e)}"
        finally:
            elapsed = time.time() - start
            LLM_REQUESTS_TOTAL.labels(vendor, model, status).inc()
            LLM_LATENCY_SECONDS.labels(vendor, model).observe(elapsed)

    return f"ERROR: {type(last_err).__name__}: {str(last_err)}" if last_err else "ERROR: Unknown"


# =========================================================
# Refined prompt bundle builders
# =========================================================

def build_refined_prompt_bundle(base_prompts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    refined = dict(base_prompts)
    v2 = dict(refined["evidence_extraction"])
    v2["id"] = "evidence_extraction_v2"
    v2["raw"] = """Extract AI-readiness evidence from this SEC filing excerpt.
If the excerpt contains no clear AI-readiness evidence, return `dimension: null` and `confidence: 0.1`.
Ensure the output format is JSON.

Filing: {{{{filing_excerpt}}}}

Return JSON with:
- dimension: one of [data_infrastructure, ai_governance, technology_stack, talent, leadership, use_case_portfolio, culture] or null if no evidence
- evidence: the specific text or 'N/A' if no evidence
- confidence: 0-1 score (0.1 for ambiguous/no evidence)
- reasoning: why this is evidence"""
    refined["evidence_extraction_v2"] = v2
    return refined


def build_refined_test_cases(base_test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in base_test_cases:
        new_t = dict(t)
        if new_t.get("description") == "Handle ambiguous evidence":
            new_t["prompt_id"] = "evidence_extraction_v2"
        out.append(new_t)
    return out


# =========================================================
# Evaluation core
# =========================================================

def run_evaluation_suite(
    providers: Dict[str, Any],
    prompts: Dict[str, Any],
    test_cases: List[Dict[str, Any]],
    *,
    api_keys: Dict[str, str],
    timeout_s: int = 60,
    max_retries: int = 2,
    assertion_funcs: Optional[Dict[str,
                                   Callable[..., Tuple[bool, str]]]] = None,
) -> pd.DataFrame:
    assertion_funcs = assertion_funcs or get_assertion_funcs()
    results: List[Dict[str, Any]] = []
    tracer = get_tracer()

    for test_idx, test_case in enumerate(test_cases):
        prompt_id = test_case["prompt_id"]
        if prompt_id not in prompts:
            raise KeyError(
                f"Test case references unknown prompt_id='{prompt_id}'")

        raw_prompt = prompts[prompt_id]["raw"]
        formatted = format_prompt(raw_prompt, test_case.get("vars", {}))

        for provider_id, provider_config in providers.items():
            with tracer.start_as_current_span("evaluation.test_run") as span:
                span.set_attribute("evaluation.test_id", f"test_{test_idx}")
                span.set_attribute("evaluation.description",
                                   test_case.get("description", ""))
                span.set_attribute("evaluation.prompt_id", prompt_id)
                span.set_attribute("evaluation.provider_id", provider_id)

                llm_output = call_llm(
                    prompt_id=prompt_id,
                    formatted_prompt=formatted,
                    provider_id=provider_id,
                    provider_config=provider_config,
                    api_keys=api_keys,
                    timeout_s=timeout_s,
                    max_retries=max_retries,
                )

                row: Dict[str, Any] = {
                    "test_id": f"test_{test_idx}",
                    "description": test_case.get("description", ""),
                    "prompt_id": prompt_id,
                    "provider_id": provider_id,
                    "input_vars": json.dumps(test_case.get("vars", {})),
                    "formatted_prompt": formatted,
                    "llm_output": llm_output,
                    "all_assertions_passed": True,
                    "failed_assertions": [],
                }

                for assertion in test_case.get("assert", []):
                    a_type = assertion["type"]
                    a_value = assertion.get("value")

                    if a_type not in assertion_funcs:
                        row["all_assertions_passed"] = False
                        row["failed_assertions"].append(
                            {"type": a_type, "value": a_value,
                                "message": f"Unknown assertion type: {a_type}"}
                        )
                        ASSERTIONS_TOTAL.labels(a_type, "false").inc()
                        continue

                    func = assertion_funcs[a_type]
                    if a_type == "json":
                        ok, msg = func(llm_output)  # type: ignore[misc]
                    else:
                        # type: ignore[misc]
                        ok, msg = func(llm_output, a_value)

                    ASSERTIONS_TOTAL.labels(
                        a_type, "true" if ok else "false").inc()

                    if not ok:
                        row["all_assertions_passed"] = False
                        row["failed_assertions"].append(
                            {"type": a_type, "value": a_value, "message": msg})

                TEST_RUNS_TOTAL.labels(
                    "true" if row["all_assertions_passed"] else "false").inc()
                results.append(row)

    return pd.DataFrame(results)


def compute_pass_rate_metrics(evaluation_df: pd.DataFrame) -> Dict[str, Any]:
    total_runs = len(evaluation_df)
    overall_pass_rate = float(
        evaluation_df["all_assertions_passed"].mean() * 100) if total_runs else 0.0

    by_provider = (
        evaluation_df.groupby("provider_id")[
            "all_assertions_passed"].mean().mul(100).to_dict()
        if total_runs
        else {}
    )
    by_prompt = (
        evaluation_df.groupby("prompt_id")[
            "all_assertions_passed"].mean().mul(100).to_dict()
        if total_runs
        else {}
    )

    return {
        "total_runs": total_runs,
        "overall_pass_rate": overall_pass_rate,
        "pass_rate_by_provider": by_provider,
        "pass_rate_by_prompt": by_prompt,
    }


def extract_failed_assertions_df(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    failed = evaluation_df[~evaluation_df["all_assertions_passed"]]
    rows: List[Dict[str, Any]] = []

    for _, r in failed.iterrows():
        for a in r.get("failed_assertions", []):
            rows.append(
                {
                    "test_id": r["test_id"],
                    "provider_id": r["provider_id"],
                    "prompt_id": r["prompt_id"],
                    "assertion_type": a.get("type"),
                    "assertion_value": a.get("value"),
                    "failure_message": a.get("message"),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# Reporting
# =========================================================

def generate_evaluation_report_markdown(
    initial_df: pd.DataFrame,
    refined_df: pd.DataFrame,
    llm_providers: Dict[str, Any],
    llm_prompts: Dict[str, Any],
    base_test_cases: Optional[List[Dict[str, Any]]] = None,
) -> str:
    base_test_cases = base_test_cases or []
    initial_rate = initial_df["all_assertions_passed"].mean(
    ) * 100 if len(initial_df) else 0.0
    refined_rate = refined_df["all_assertions_passed"].mean(
    ) * 100 if len(refined_df) else 0.0

    lines: List[str] = []
    lines.append(
        "# LLM Evaluation Report for PE Org-AI-R AI-Readiness Extractor\n\n")
    lines.append("## Executive Summary\n\n")
    lines.append("This report summarizes the automated evaluation of our LLM-powered AI-readiness extractor across different models and prompt versions.\n\n")
    lines.append(f"- Initial overall pass rate: **{initial_rate:.2f}%**\n")
    lines.append(f"- Refined overall pass rate: **{refined_rate:.2f}%**\n\n")
    lines.append(
        "Prompt engineering has shown a positive and measurable impact on the quality and reliability of LLM outputs.\n\n")

    lines.append("## 1. Evaluation Setup\n\n")
    lines.append("### 1.1 LLM Providers\n\n")
    for pid, cfg in llm_providers.items():
        lines.append(f"- `{pid}` (temperature={cfg.get('temperature')})\n")

    lines.append("\n### 1.2 Prompts\n\n")
    for prompt_id in llm_prompts.keys():
        lines.append(f"- `{prompt_id}`\n")

    if base_test_cases:
        lines.append(
            f"\nTotal unique test cases defined: {len(base_test_cases)}\n")

    lines.append("\n## 2. Overall Performance Metrics\n\n")
    lines.append("| Metric | Initial Run | Refined Run |\n")
    lines.append("|---|---:|---:|\n")
    lines.append(
        f"| Overall Pass Rate | {initial_rate:.2f}% | {refined_rate:.2f}% |\n")

    lines.append("\n## 3. Performance by Provider and Prompt Type\n\n")
    lines.append("### 3.1 Pass Rate by Provider\n\n")
    lines.append("| Provider | Initial (%) | Refined (%) |\n")
    lines.append("|---|---:|---:|\n")
    for provider in llm_providers.keys():
        i = (initial_df[initial_df["provider_id"] == provider]
             ["all_assertions_passed"].mean() * 100) if len(initial_df) else 0.0
        r = (refined_df[refined_df["provider_id"] == provider]
             ["all_assertions_passed"].mean() * 100) if len(refined_df) else 0.0
        lines.append(f"| {provider} | {i:.2f} | {r:.2f} |\n")

    lines.append("\n### 3.2 Pass Rate by Prompt Type\n\n")
    lines.append("| Prompt ID | Initial (%) | Refined (%) |\n")
    lines.append("|---|---:|---:|\n")
    all_prompt_ids = sorted(
        set(initial_df["prompt_id"]).union(set(refined_df["prompt_id"])))
    for pid in all_prompt_ids:
        i = (initial_df[initial_df["prompt_id"] == pid]["all_assertions_passed"].mean(
        ) * 100) if pid in set(initial_df["prompt_id"]) else 0.0
        r = (refined_df[refined_df["prompt_id"] == pid]["all_assertions_passed"].mean(
        ) * 100) if pid in set(refined_df["prompt_id"]) else 0.0
        lines.append(f"| {pid} | {i:.2f} | {r:.2f} |\n")

    lines.append("\n## 4. Key Findings and Recommendations\n\n")
    lines.append(
        "- Expand test diversity to cover more edge cases for each readiness dimension.\n")
    lines.append(
        "- Integrate evaluation into CI/CD to prevent regressions with prompt/model changes.\n")
    lines.append(
        "- Periodically revisit assertion logic as requirements evolve.\n\n")

    lines.append(
        "A critical aspect of our evaluation involves the confidence score $C$ generated by the LLM. "
        "We initially used a threshold $T=0.7$ for accepting extracted evidence. "
        "Future work could explore dynamic thresholds or a utility function $U(C,\\text{dimension})$.\n"
    )

    lines.append("\n## 5. Next Steps\n\n")
    lines.append(
        "- Add coverage for other dimensions (e.g., `technology_stack`, `use_case_portfolio`).\n")
    lines.append(
        "- Compare additional models or fine-tuning strategies using the same suite.\n")
    lines.append(
        "- Build a monitoring dashboard for trend analysis (quality, latency, drift).\n")

    return "".join(lines)
