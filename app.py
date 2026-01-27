# app.py
import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Callable, Tuple
import os
from source import *

st.set_page_config(
    page_title="QuLab: Lab 14: Production Hardening & Governance", layout="wide")

# --- Observability bootstrap (safe to call multiple times) ---
if "otel_initialized" not in st.session_state:
    # You can optionally set an OTLP endpoint via environment variable for real deployments.
    # Example: export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    init_tracing(service_name="qulab-lab14",
                 otlp_endpoint=otlp_endpoint, enable_console_exporter=True)
    st.session_state.otel_initialized = True

if "prometheus_started" not in st.session_state:
    # Prometheus scrape endpoint: http://<host>:8000/metrics
    # Configure your Prometheus to scrape it; Grafana can visualize via Prometheus datasource.
    start_prometheus_metrics_server(
        port=int(os.getenv("PROMETHEUS_PORT", "8000")))
    st.session_state.prometheus_started = True

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 14: Production Hardening & Governance")
st.divider()

# --- Sidebar: API Keys & Observability ---
st.sidebar.subheader("LLM Credentials")
openai_key_input = st.sidebar.text_input(
    "OpenAI API Key", type="password", value=st.session_state.get("openai_api_key", ""))
st.session_state.openai_api_key = openai_key_input.strip()


# --- Session State Initialization ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Introduction & Setup'
if 'llm_providers' not in st.session_state:
    st.session_state.llm_providers = DEFAULT_LLM_PROVIDERS
if 'llm_prompts' not in st.session_state:
    st.session_state.llm_prompts = DEFAULT_LLM_PROMPTS
if 'test_cases' not in st.session_state:
    st.session_state.test_cases = DEFAULT_TEST_CASES
if 'evaluation_results_df' not in st.session_state:
    st.session_state.evaluation_results_df = None
if 'refined_llm_prompts' not in st.session_state:
    st.session_state.refined_llm_prompts = None
if 'refined_test_cases' not in st.session_state:
    st.session_state.refined_test_cases = None
if 're_evaluation_results_df' not in st.session_state:
    st.session_state.re_evaluation_results_df = None

# --- Sidebar Navigation ---
st.sidebar.divider()
st.sidebar.title("Navigation")
page_options = [
    "Introduction & Setup",
    "Define Prompts & Test Cases",
    "Run Evaluation",
    "Analyze Results",
    "Iterate & Refine",
    "Final Report & Regression"
]

# Ensure current_page is valid
if st.session_state.current_page not in page_options:
    st.session_state.current_page = page_options[0]

st.session_state.current_page = st.sidebar.selectbox(
    "Choose a section:",
    page_options,
    index=page_options.index(st.session_state.current_page)
)


st.sidebar.markdown("""
---
### Learning Objectives

* **Remember**: List observability pillars (metrics, logs, traces)
* **Understand**: Explain why LLM evaluation is critical
* **Apply**: Implement **promptfoo** evaluation suite
* **Analyze**: Debug agent workflows with traces
* **Create**: Design production monitoring system

---

### Tools Introduced

* **promptfoo**: LLM evaluation
* **OpenTelemetry**: Tracing
* **Prometheus**: Metrics
* **Grafana**: Dashboards

---
                    """)


# --- Main Content Area ---

# --- Page: Introduction & Setup ---
if st.session_state.current_page == "Introduction & Setup":
    st.markdown(
        f"## 1. Introduction: Ensuring AI Readiness with Robust LLM Evaluation")
    st.markdown(f"As a **Software Developer** at **PE Org-AI-R**, my team is building a critical AI-powered feature: an AI-readiness extractor. This tool analyzes SEC filings and other corporate documents to identify evidence of a company's readiness across various AI dimensions like `data_infrastructure`, `ai_governance`, and `technology_stack`. Before we deploy this feature or update its underlying Large Language Models (LLMs), I need to guarantee that its outputs are consistently high-quality, accurate, and safe. Traditional manual testing is slow and doesn't scale, making automated evaluation indispensable.")
    st.markdown(f"This application outlines a real-world workflow to implement an automated LLM evaluation suite, conceptually similar to `promptfoo`, to systematically assess our LLM's performance. My goal is to rapidly identify issues, compare different LLM configurations, and ultimately ship features with confidence.")

# --- Page: Define Prompts & Test Cases ---
elif st.session_state.current_page == "Define Prompts & Test Cases":
    st.markdown(f"## 3. Defining LLM Providers and Core Prompts")
    st.markdown(f"To begin our evaluation, I need to define the different LLM providers and models we're considering, along with the core prompts our AI-readiness extractor will use. For PE Org-AI-R, we're currently exploring `OpenAI`'s `gpt-4o` and `Anthropic`'s `claude-sonnet`. Each model will have specific configuration parameters, such as `temperature`, which influences the creativity and randomness of the LLM's output.")
    st.markdown(
        f"This section simulates the `providers` and `prompts` sections of a `promptfoo/config.yaml`.")

    st.subheader("Current LLM Providers")
    providers_df = (
        pd.DataFrame.from_dict(st.session_state.llm_providers, orient="index")
        .reset_index()
        .rename(columns={"index": "provider_id"})
    )
    st.dataframe(
        providers_df,
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Edit providers in code/config; this table is read-only in the UI.")

    st.subheader("Current LLM Prompts")

    prompts_items = []
    for pid, p in st.session_state.llm_prompts.items():
        raw = p.get("raw", "")
        prompts_items.append(
            {
                "prompt_id": pid,
                "template_preview": (raw[:180] + "…") if len(raw) > 180 else raw,
                "template_chars": len(raw),
            }
        )
    prompts_df = pd.DataFrame(prompts_items).sort_values("prompt_id")

    col_a, col_b, col_c = st.columns([2, 6, 2])
    with col_a:
        selected_prompt_id = st.selectbox(
            "Select a prompt to inspect",
            prompts_df["prompt_id"].tolist(),
            index=0 if len(prompts_df) else None,
            key="selected_prompt_id",
        )
    with col_b:
        st.dataframe(prompts_df, use_container_width=True, hide_index=True)
    with col_c:
        st.metric("Total Prompts", int(len(prompts_df)))

    if selected_prompt_id:
        st.markdown("### Prompt Template")
        st.code(
            st.session_state.llm_prompts[selected_prompt_id]["raw"], language="text")

    st.markdown(f"## 4. Crafting Diverse Test Cases and Assertions")
    st.markdown(f"A robust evaluation starts with diverse test cases that cover critical scenarios. As a Software Developer, I'm responsible for defining these tests, including the input variables (`vars`) and the `assert` conditions that define a \"pass.\" These assertions act as our ground truth, verifying the LLM's output against expected formats, content, and quality metrics.")
    st.markdown(
        f"This section simulates the `tests` section of a `promptfoo/config.yaml`.")

    st.subheader("Current Test Cases")
    if st.button("Load Default Configuration"):
        st.session_state.llm_providers = DEFAULT_LLM_PROVIDERS
        st.session_state.llm_prompts = DEFAULT_LLM_PROMPTS
        st.session_state.test_cases = DEFAULT_TEST_CASES
        st.success("Default LLM Providers, Prompts, and Test Cases loaded!")

    # --- REPLACE #3: st.json(st.session_state.test_cases) (after Load Default Configuration button) ---
    tests_items = []
    for i, t in enumerate(st.session_state.test_cases):
        vars_dict = t.get("vars", {}) or {}
        asserts_list = t.get("assert", []) or []
        tests_items.append(
            {
                "test_idx": i,
                "description": t.get("description", ""),
                "prompt_id": t.get("prompt_id", ""),
                "vars_keys": ", ".join(list(vars_dict.keys())) if vars_dict else "",
                "assertions_count": len(asserts_list),
            }
        )
    tests_df = pd.DataFrame(tests_items)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_test_idx = st.selectbox(
            "Select a test case to inspect",
            tests_df["test_idx"].tolist(),
            format_func=lambda x: f"[{x}] {tests_df.loc[tests_df['test_idx'] == x, 'description'].values[0]}",
            key="selected_test_idx",
        )
    with col2:
        st.metric("Total Test Cases", int(len(tests_df)))

    st.dataframe(tests_df, use_container_width=True, hide_index=True)

    if selected_test_idx is not None:
        tc = st.session_state.test_cases[int(selected_test_idx)]
        st.markdown("### Test Case Details")
        a, b = st.columns([1, 1])
        with a:
            st.markdown("**Vars**")
            st.dataframe(
                pd.DataFrame([{"key": k, "value": str(v)}
                             for k, v in (tc.get("vars", {}) or {}).items()]),
                use_container_width=True,
                hide_index=True,
            )
        with b:
            st.markdown("**Assertions**")
            st.dataframe(
                pd.DataFrame(tc.get("assert", []) or []),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Prompt Preview (rendered)**")
        prompt_cfg = st.session_state.llm_prompts.get(
            tc.get("prompt_id", ""), {})
        rendered = format_prompt(prompt_cfg.get(
            "raw", ""), tc.get("vars", {}) or {})
        st.code(rendered, language="text")

    st.markdown(f"### Explanation of Test Case Design")
    st.markdown(f"Each test case consists of:")
    st.markdown(
        f"*   A `description` for human readability, explaining the scenario.")
    st.markdown(
        f"*   A `prompt_id` linking to one of our predefined LLM prompts (e.g., `evidence_extraction`).")
    st.markdown(
        f"*   `vars`: A dictionary of input variables that will populate the prompt template, making the prompt dynamic.")
    st.markdown(f"*   `assert`: A list of conditions that the LLM's output must satisfy. These assertions are critical for defining what a \"good\" output looks like. They can check for JSON validity (`type: json`), specific content presence (`type: contains`), or complex logic simulated by `type: javascript`.")
    st.markdown(r"For example, for an evidence extraction task, we might expect a confidence score $C$ for the extracted evidence to be above a certain threshold $T$, represented mathematically as $$ C \ge T $$")
    st.markdown(r"where $C$ is the confidence score (0-1) provided by the LLM, and $T$ is the minimum acceptable threshold (e.g., 0.7).")
    st.markdown(
        f"This helps ensure the reliability of the extracted information.")

# --- Page: Run Evaluation ---
elif st.session_state.current_page == "Run Evaluation":
    st.markdown(f"## 5. Running the Automated Evaluation Suite")
    st.markdown(f"Now that our test cases and assertion logic are defined, I can run the automated evaluation suite. This involves iterating through each test case, dynamically rendering the prompt with the specified variables, \"calling\" each LLM provider, and then applying all defined assertions to the LLM's response. This systematic approach allows us to quickly assess how different models perform across various scenarios.")
    st.markdown(f"We will simulate LLM responses to ensure reproducibility and avoid external API calls for this lab. The `mock_llm_response` function will generate plausible, structured responses based on the input prompt and test case.")

    if st.button("Run Initial Evaluation"):
        if not st.session_state.openai_api_key:
            st.error(
                "Please enter your OpenAI API Key in the sidebar to run OpenAI providers.")
        with st.spinner("Running evaluation..."):
            st.session_state.evaluation_results_df = run_evaluation_suite(
                st.session_state.llm_providers,
                st.session_state.llm_prompts,
                st.session_state.test_cases,
                api_keys={"openai": st.session_state.openai_api_key},
                timeout_s=60,
                max_retries=2,
            )
        st.success("Evaluation complete!")

    if st.session_state.evaluation_results_df is not None:
        st.subheader("Evaluation Results")
        st.dataframe(st.session_state.evaluation_results_df)

        st.markdown(f"### Explanation of Execution Results")
        st.markdown(f"The `evaluation_results_df` DataFrame now contains the results for each test case run against each LLM provider. Each row represents a specific `test_id` (input scenario) and `provider_id` (model). The `all_assertions_passed` column immediately tells me if the LLM output met all the predefined quality criteria for that specific test. The `failed_assertions` column provides details on why a test might have failed, including the type of assertion and the specific message. This granular data is crucial for debugging and understanding performance differences between models and prompt versions. It allows me to pinpoint exactly where the LLM's output deviates from our expectations.")
    else:
        st.info("Run the initial evaluation to see results.")

# --- Page: Analyze Results ---
elif st.session_state.current_page == "Analyze Results":
    st.markdown(
        f"## 6. Analyzing Performance: Aggregated Metrics and Failure Patterns")
    st.markdown(f"After running the evaluation, I need to analyze the results to understand the LLM's overall performance. This involves looking at aggregated metrics like pass/fail rates and drilling down into specific failure patterns. This step helps me quickly identify which LLMs are performing best and which areas of our AI-readiness extractor (e.g., specific dimensions or types of evidence) need improvement. Visualizations are key here to quickly grasp the larger picture.")

    if st.session_state.evaluation_results_df is not None:
        # Aggregated Metrics
        total_tests_runs = len(st.session_state.evaluation_results_df)
        overall_pass_rate = st.session_state.evaluation_results_df["all_assertions_passed"].mean(
        ) * 100

        st.subheader("Overall Performance Metrics")
        st.metric("Overall Pass Rate Across All Tests and Providers",
                  f"{overall_pass_rate:.2f}%")

        st.subheader("Pass Rate by Provider")
        pass_rate_by_provider = st.session_state.evaluation_results_df.groupby(
            "provider_id")["all_assertions_passed"].mean() * 100
        st.dataframe(pass_rate_by_provider.apply(lambda x: f"{x:.2f}%"))

        st.subheader("Pass Rate by Prompt ID")
        pass_rate_by_prompt = st.session_state.evaluation_results_df.groupby(
            "prompt_id")["all_assertions_passed"].mean() * 100
        st.dataframe(pass_rate_by_prompt.apply(lambda x: f"{x:.2f}%"))

        # Visualization of Pass Rates
        st.subheader("Visualizations")
        # Create a figure and a set of subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        pass_rate_by_provider.plot(
            kind="bar", color=["skyblue", "lightcoral"], ax=axes[0])
        axes[0].set_title("Pass Rate by LLM Provider")
        axes[0].set_ylabel("Pass Rate (%)")
        axes[0].set_ylim(0, 100)
        axes[0].tick_params(axis='x', rotation=45)

        pass_rate_by_prompt.plot(
            kind="bar", color=["lightgreen", "salmon"], ax=axes[1])
        axes[1].set_title("Pass Rate by Prompt Type")
        axes[1].set_ylabel("Pass Rate (%)")
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        st.pyplot(fig)  # Pass the figure object to st.pyplot

        # Analysis of Failure Patterns
        st.subheader("Analysis of Failure Patterns")
        failed_tests_df = st.session_state.evaluation_results_df[
            ~st.session_state.evaluation_results_df["all_assertions_passed"]]
        st.markdown(f"Total Failed Test Runs: {len(failed_tests_df)}")

        if not failed_tests_df.empty:
            all_failed_assertions = []
            for _, row in failed_tests_df.iterrows():
                for assertion in row["failed_assertions"]:
                    all_failed_assertions.append({
                        "test_id": row["test_id"],
                        "provider_id": row["provider_id"],
                        "prompt_id": row["prompt_id"],
                        "assertion_type": assertion["type"],
                        "assertion_value": assertion["value"],
                        "failure_message": assertion["message"]
                    })
            failed_assertions_df = pd.DataFrame(all_failed_assertions)

            st.markdown(f"### Top Failing Assertion Types")
            failing_assertion_counts = failed_assertions_df["assertion_type"].value_counts(
            )
            st.dataframe(failing_assertion_counts)

            fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
            failing_assertion_counts.plot(
                kind="pie", autopct='%1.1f%%', startangle=90, cmap='Pastel1', ax=ax_pie)
            ax_pie.set_title("Distribution of Failing Assertion Types")
            # Hide the default 'count' label for pie chart
            ax_pie.set_ylabel("")
            st.pyplot(fig_pie)

            st.markdown(f"### Examples of Failed Test Runs")
            # Display failed tests using st.expander for better readability
            # Limiting to first 5 for brevity
            for i, row in failed_tests_df.head(5).iterrows():
                with st.expander(f"Failed Test: {row['description']} ({row['provider_id']})"):
                    # --- REPLACE #4: st.json({"LLM Output": row['llm_output'], "Failed Assertions": row['failed_assertions']}) ---
                    st.markdown("**LLM Output**")
                    st.code(str(row["llm_output"]), language="text")

                    st.markdown("**Failed Assertions**")
                    failed_df = pd.DataFrame(row["failed_assertions"] or [])
                    if failed_df.empty:
                        st.success(
                            "No failed assertions recorded for this run.")
                    else:
                        st.dataframe(
                            failed_df, use_container_width=True, hide_index=True)

        else:
            st.success(
                "No test runs failed in this initial run. Excellent work!")

        st.markdown(f"### Explanation of Execution")
        st.markdown(f"The visualizations clearly show the `Pass Rate by LLM Provider` and `Pass Rate by Prompt Type`. This helps me understand which models are more reliable and which types of tasks (e.g., `evidence_extraction` vs. `dimension_scoring`) are more challenging for our current prompts and models. The `Distribution of Failing Assertion Types` pie chart pinpoints common failure modes. For instance, if `javascript` assertions (which often check numerical conditions like confidence scores or extracted values) are failing more often than `json` structure checks, it signals a specific type of problem.")
        st.markdown(f"For example, if `output.confidence >= 0.7` is frequently failing, it tells me the LLM is not consistently providing high-confidence evidence, or our threshold $T=0.7$ for acceptable confidence is too strict for the model's capabilities in certain scenarios. Adjusting the prompt to encourage higher confidence outputs or re-evaluating the threshold $T$ could be next steps.")
        st.markdown(
            r"The pass rate $P$ is calculated as:$$ P = \frac{{\text{{Number of Passed Tests}}}}{{\text{{Total Number of Tests}}}} \times 100\% $$")
        st.markdown(r"where 'Number of Passed Tests' is the count of test runs where `all_assertions_passed` is `True`, and 'Total Number of Tests' is the total number of evaluation runs. This metric provides a high-level overview of the system's performance for easy comparison.")
    else:
        st.info(
            "Please run the initial evaluation first on the 'Run Evaluation' page to analyze results.")

# --- Page: Iterate & Refine ---
elif st.session_state.current_page == "Iterate & Refine":
    st.markdown(
        f"## 7. Iterative Improvement: Prompt Engineering and Threshold Adjustment")
    st.markdown(f"Identifying failures is the first step; the next is to iterate and improve. As a Software Developer, I'll use the failure analysis to refine our prompts and possibly adjust assertion thresholds. For instance, if the \"Handle ambiguous evidence\" test consistently fails because the confidence isn't low enough, I might need to clarify the prompt's instructions for uncertain scenarios or adjust the confidence threshold $T$.")
    st.markdown(f"Let's simulate a prompt refinement and re-evaluation for a specific failing test. We noticed our \"Handle ambiguous evidence\" test might be tricky. The original assertion was `output.confidence < 0.5 || output.dimension === null`. We could refine the prompt to explicitly instruct the LLM to return `null` for the dimension and a very low confidence if evidence is ambiguous, thereby improving its adherence to our defined quality criteria.")

    if st.session_state.evaluation_results_df is not None:
        if st.button("Simulate Prompt Refinement and Re-evaluate"):
            # Simulate Prompt Refinement (from source.py logic)
            st.session_state.refined_llm_prompts = build_refined_prompt_bundle(
                st.session_state.llm_prompts)
            st.session_state.refined_test_cases = build_refined_test_cases(
                st.session_state.test_cases)

            with st.spinner("Running re-evaluation with refined prompt..."):
                st.session_state.re_evaluation_results_df = run_evaluation_suite(
                    st.session_state.llm_providers,
                    st.session_state.refined_llm_prompts,  # Use refined prompts
                    st.session_state.refined_test_cases,  # Use refined test cases
                    api_keys={"openai": st.session_state.openai_api_key},
                    timeout_s=60,
                    max_retries=2,
                )
            st.success("Re-evaluation with refined prompt complete!")

        if st.session_state.re_evaluation_results_df is not None:
            st.subheader(
                "Results for 'Handle ambiguous evidence' with refined prompt:")
            ambiguous_test_results_refined = st.session_state.re_evaluation_results_df[
                (st.session_state.re_evaluation_results_df["description"] == "Handle ambiguous evidence") &
                (st.session_state.re_evaluation_results_df["prompt_id"]
                 == "evidence_extraction_v2")
            ]
            st.dataframe(ambiguous_test_results_refined[[
                         "provider_id", "all_assertions_passed", "failed_assertions"]])

            st.subheader("Original results for 'Handle ambiguous evidence':")
            original_ambiguous_test_results = st.session_state.evaluation_results_df[
                (st.session_state.evaluation_results_df["description"] == "Handle ambiguous evidence") &
                (st.session_state.evaluation_results_df["prompt_id"]
                 == "evidence_extraction")
            ]
            st.dataframe(original_ambiguous_test_results[[
                         "provider_id", "all_assertions_passed", "failed_assertions"]])

            st.subheader("Overall Pass Rate Comparison")
            overall_pass_rate_initial = st.session_state.evaluation_results_df["all_assertions_passed"].mean(
            ) * 100
            overall_pass_rate_refined = st.session_state.re_evaluation_results_df["all_assertions_passed"].mean(
            ) * 100

            comparison_df = pd.DataFrame({
                'Run': ['Initial Evaluation', 'Refined Prompt Evaluation'],
                'Pass Rate (%)': [overall_pass_rate_initial, overall_pass_rate_refined]
            })

            fig_comp, ax_comp = plt.subplots(figsize=(7, 5))
            sns.barplot(x='Run', y='Pass Rate (%)',
                        data=comparison_df, palette='viridis', ax=ax_comp)
            ax_comp.set_title('Overall Pass Rate Comparison')
            ax_comp.set_ylim(0, 100)
            ax_comp.set_ylabel('Pass Rate (%)')
            st.pyplot(fig_comp)

            st.markdown(
                f"Overall pass rate increased from {overall_pass_rate_initial:.2f}% to {overall_pass_rate_refined:.2f}% due to prompt refinement.")
        else:
            st.info(
                "Click the button above to simulate prompt refinement and run re-evaluation.")
    else:
        st.info(
            "Please run the initial evaluation first on the 'Run Evaluation' page before iterating.")

# --- Page: Final Report & Regression ---
elif st.session_state.current_page == "Final Report & Regression":
    st.markdown(
        f"## 8. Tracking Regressions and Generating the Evaluation Report")
    st.markdown(f"To prevent regressions with future deployments, it's crucial to track evaluation scores over time or across different model or prompt versions. This allows us to ensure that any new changes maintain or improve quality consistently. Finally, I'll generate a comprehensive 'LLM Evaluation Report' to summarize our findings for stakeholders, providing objective data for decision-making.")

    if st.session_state.evaluation_results_df is not None and st.session_state.re_evaluation_results_df is not None:
        st.subheader("Pass Rate Comparison Across Evaluation Versions")
        # Add 'version' column for easier comparison
        initial_results_snapshot = st.session_state.evaluation_results_df.copy()
        refined_results_snapshot = st.session_state.re_evaluation_results_df.copy()

        initial_results_snapshot['version'] = 'Initial'
        refined_results_snapshot['version'] = 'Refined Prompt'

        comparison_df_long = pd.concat(
            [initial_results_snapshot, refined_results_snapshot])

        pass_rates_comparison = comparison_df_long.groupby(['version', 'provider_id'])[
            'all_assertions_passed'].mean().unstack() * 100
        st.dataframe(pass_rates_comparison.applymap(lambda x: f"{x:.2f}%"))

        fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
        pass_rates_comparison.plot(kind='bar', colormap='viridis', ax=ax_reg)
        ax_reg.set_title('Pass Rate by Provider Across Evaluation Versions')
        ax_reg.set_ylabel('Pass Rate (%)')
        ax_reg.set_ylim(0, 100)
        ax_reg.tick_params(axis='x', rotation=0)
        ax_reg.legend(title='Provider')
        ax_reg.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_reg)

        st.markdown(f"### Generating the LLM Evaluation Report")
        # Start with original defaults
        combined_llm_prompts = {**DEFAULT_LLM_PROMPTS}
        if st.session_state.refined_llm_prompts:
            # Add/overwrite with refined ones
            combined_llm_prompts.update(st.session_state.refined_llm_prompts)

        llm_evaluation_report = generate_evaluation_report_markdown(
            initial_df=st.session_state.evaluation_results_df,
            refined_df=st.session_state.re_evaluation_results_df,
            llm_providers=st.session_state.llm_providers,
            llm_prompts=combined_llm_prompts,
            base_test_cases=st.session_state.test_cases
        )
        st.markdown(llm_evaluation_report)

        st.markdown(f"### Explanation of Results and Report")
        st.markdown(f"The \"Pass Rate Comparison Across Evaluation Versions\" visualization is a clear example of **Regression Analysis**. It shows how the performance of each LLM provider changed from the 'Initial' evaluation to the 'Refined Prompt' evaluation. Ideally, all bars in the 'Refined Prompt' section should be equal to or higher than their 'Initial' counterparts, indicating no regressions and overall improvement. This objective, quantitative comparison is vital for validating changes.")
        st.markdown(f"The `LLM Evaluation Report` provides a structured, detailed summary for stakeholders. It covers the evaluation setup, comparative performance metrics across different LLM providers and prompt types, key findings from the analysis, and actionable recommendations for future development. This report is our deliverable, allowing the team to make informed decisions about LLM selection, prompt strategies, and deployment readiness, backed by objective data. It emphasizes the importance of automated evaluation for maintaining and continuously improving the quality of our AI-powered features, ensuring they meet the high standards expected by PE Org-AI-R.")
    else:
        st.info("Please run both the initial and refined evaluations to generate the full report and regression analysis.")

# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
