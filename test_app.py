
import pandas as pd
from streamlit.testing.v1 import AppTest
import sys

# --- Dummy data for consistent testing ---
DUMMY_LLM_PROVIDERS = {
    "openai": {"model": "gpt-4o", "api_key_env": "OPENAI_API_KEY", "temperature": 0.7},
    "anthropic": {"model": "claude-sonnet", "api_key_env": "ANTHROPIC_API_KEY", "temperature": 0.5}
}

DUMMY_LLM_PROMPTS = {
    "evidence_extraction": {
        "id": "evidence_extraction",
        "raw": "Extract AI-readiness evidence from: {{filing_excerpt}}"
    }
}

DUMMY_TEST_CASES = [
    {
        "description": "Extract data infrastructure evidence",
        "prompt_id": "evidence_extraction",
        "vars": {"filing_excerpt": "Our robust data infrastructure supports AI initiatives."},
        "assert": [{"type": "json"}, {"type": "contains", "value": "data_infrastructure"}]
    },
    {
        "description": "Handle ambiguous evidence",
        "prompt_id": "evidence_extraction",
        "vars": {"filing_excerpt": "We are exploring new technologies and nothing concrete."},
        "assert": [{"type": "json"}, {"type": "javascript", "value": "output.confidence < 0.5 || output.dimension === null"}]
    }
]

# Create dummy evaluation results DataFrames that reflect the expected pass/fail logic
DUMMY_EVALUATION_RESULTS_DF = pd.DataFrame([
    {
        "test_id": "0",
        "description": "Extract data infrastructure evidence",
        "prompt_id": "evidence_extraction",
        "provider_id": "openai",
        "llm_output": '{"dimension": "data_infrastructure", "evidence": "Our robust data infrastructure supports AI initiatives.", "confidence": 0.9, "reasoning": "Explicit mention of data infrastructure."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    },
    {
        "test_id": "0",
        "description": "Extract data infrastructure evidence",
        "prompt_id": "evidence_extraction",
        "provider_id": "anthropic",
        "llm_output": '{"dimension": "data_infrastructure", "evidence": "Our robust data infrastructure supports AI initiatives.", "confidence": 0.8, "reasoning": "Explicit mention."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    },
    {
        "test_id": "1",
        "description": "Handle ambiguous evidence",
        "prompt_id": "evidence_extraction",
        "provider_id": "openai",
        "llm_output": '{"dimension": null, "evidence": "N/A", "confidence": 0.2, "reasoning": "No clear evidence for ambiguous input."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    },
    {
        "test_id": "1",
        "description": "Handle ambiguous evidence",
        "prompt_id": "evidence_extraction",
        "provider_id": "anthropic",
        "llm_output": '{"dimension": "technology_stack", "evidence": "exploring new technologies", "confidence": 0.6, "reasoning": "Broad statement, might be technology_stack."}',
        "all_assertions_passed": False,
        "failed_assertions": [{"type": "javascript", "value": "output.confidence < 0.5 || output.dimension === null", "message": "Assertion failed: confidence not low enough and dimension not null"}],
    }
])

DUMMY_RE_EVALUATION_RESULTS_DF = pd.DataFrame([
    {
        "test_id": "0",
        "description": "Extract data infrastructure evidence",
        "prompt_id": "evidence_extraction",
        "provider_id": "openai",
        "llm_output": '{"dimension": "data_infrastructure", "evidence": "Our robust data infrastructure supports AI initiatives.", "confidence": 0.95, "reasoning": "Explicit mention of data infrastructure."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    },
    {
        "test_id": "0",
        "description": "Extract data infrastructure evidence",
        "prompt_id": "evidence_extraction",
        "provider_id": "anthropic",
        "llm_output": '{"dimension": "data_infrastructure", "evidence": "Our robust data infrastructure supports AI initiatives.", "confidence": 0.85, "reasoning": "Explicit mention."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    },
    {
        "test_id": "1",
        "description": "Handle ambiguous evidence",
        "prompt_id": "evidence_extraction_v2",
        "provider_id": "openai",
        "llm_output": '{"dimension": null, "evidence": "N/A", "confidence": 0.1, "reasoning": "Prompt instructed for ambiguous, very low confidence."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    },
    {
        "test_id": "1",
        "description": "Handle ambiguous evidence",
        "prompt_id": "evidence_extraction_v2",
        "provider_id": "anthropic",
        "llm_output": '{"dimension": null, "evidence": "N/A", "confidence": 0.15, "reasoning": "Prompt instructed for ambiguous, very low confidence."}',
        "all_assertions_passed": True,
        "failed_assertions": [],
    }
])

# --- Mocking the 'source' module ---
class MockSourceModule:
    def __init__(self):
        self.LLM_PROVIDERS = DUMMY_LLM_PROVIDERS
        self.LLM_PROMPTS = DUMMY_LLM_PROMPTS
        self.TEST_CASES = DUMMY_TEST_CASES

    def run_evaluation_suite(self, llm_providers, llm_prompts, test_cases, mock_response_func):
        # Determine whether to return initial or refined results based on prompt IDs
        if any("evidence_extraction_v2" in p_id for p_id in llm_prompts.keys()):
            return DUMMY_RE_EVALUATION_RESULTS_DF
        return DUMMY_EVALUATION_RESULTS_DF

    def mock_llm_response(self, prompt_id: str, prompt_vars: dict, provider_id: str, llm_prompts: dict) -> dict:
        # A minimal mock matching expected outputs for DUMMY_EVALUATION_RESULTS_DF
        if prompt_id == "evidence_extraction":
            if "data_infrastructure" in prompt_vars.get("filing_excerpt", ""):
                return {'dimension': 'data_infrastructure', 'evidence': '...', 'confidence': 0.9, 'reasoning': '...'}
            elif "ambiguous" in prompt_vars.get("filing_excerpt", ""):
                if provider_id == "openai":
                    return {'dimension': None, 'evidence': 'N/A', 'confidence': 0.2, 'reasoning': 'No clear evidence for ambiguous input.'}
                else:
                    return {'dimension': 'technology_stack', 'evidence': 'exploring new technologies', 'confidence': 0.6, 'reasoning': 'Broad statement, might be technology_stack.'}
        return {'dimension': 'unknown', 'evidence': 'N/A', 'confidence': 0.5, 'reasoning': 'Default mock response.'}

    def mock_llm_response_v2(self, prompt_id: str, prompt_vars: dict, provider_id: str, llm_prompts: dict) -> dict:
        # Mock for the refined prompt
        if prompt_id == "evidence_extraction_v2":
            return {'dimension': None, 'evidence': 'N/A', 'confidence': 0.1, 'reasoning': 'Ambiguous per v2 prompt.'}
        return self.mock_llm_response(prompt_id, prompt_vars, provider_id, llm_prompts)

    def generate_evaluation_report(self, initial_df: pd.DataFrame, refined_df: pd.DataFrame, llm_providers: dict, llm_prompts: dict) -> str:
        initial_pass_rate = initial_df['all_assertions_passed'].mean() * 100
        refined_pass_rate = refined_df['all_assertions_passed'].mean() * 100
        return (
            "## LLM Evaluation Report\n\nSummary of findings for PE Org-AI-R project.\n\n"
            f"Initial evaluation pass rate: {initial_pass_rate:.2f}%\n"
            f"Refined evaluation pass rate: {refined_pass_rate:.2f}%"
        )

# Place the mock module into sys.modules so 'app.py' imports our mock
sys.modules['source'] = MockSourceModule()

# --- Test Functions ---

def test_initial_load_and_introduction_page():
    """
    Tests if the app loads correctly and the initial page is 'Introduction & Setup'.
    Verifies the main title and introduction text.
    """
    at = AppTest.from_file("app.py").run()
    assert at.title[0].value == "QuLab: Lab 14: Production Hardening & Governance"
    assert "Introduction: Ensuring AI Readiness with Robust LLM Evaluation" in at.markdown[2].value
    assert at.session_state["current_page"] == "Introduction & Setup"


def test_navigation_to_all_pages():
    """
    Tests navigation between all pages using the sidebar selectbox.
    Verifies that the `current_page` in session state updates and a unique element
    from each page's content is present.
    """
    at = AppTest.from_file("app.py").run()

    page_options = [
        "Introduction & Setup",
        "Define Prompts & Test Cases",
        "Run Evaluation",
        "Analyze Results",
        "Iterate & Refine",
        "Final Report & Regression"
    ]

    for i, page in enumerate(page_options):
        at.sidebar.selectbox[0].set_value(page).run()
        assert at.session_state["current_page"] == page
        # Check for a unique element on each page to confirm navigation
        if page == "Introduction & Setup":
            assert "Introduction: Ensuring AI Readiness with Robust LLM Evaluation" in at.markdown[2].value
        elif page == "Define Prompts & Test Cases":
            assert "Defining LLM Providers and Core Prompts" in at.markdown[2].value
            assert "Current LLM Providers" in at.subheader[0].value
        elif page == "Run Evaluation":
            assert "Running the Automated Evaluation Suite" in at.markdown[2].value
            assert at.button[0].label == "Run Initial Evaluation"
        elif page == "Analyze Results":
            assert "Analyzing Performance: Aggregated Metrics and Failure Patterns" in at.markdown[2].value
        elif page == "Iterate & Refine":
            assert "Iterative Improvement: Prompt Engineering and Threshold Adjustment" in at.markdown[2].value
        elif page == "Final Report & Regression":
            assert "Tracking Regressions and Generating the Evaluation Report" in at.markdown[2].value


def test_define_prompts_and_test_cases_page():
    """
    Tests the "Define Prompts & Test Cases" page.
    Verifies initial display of LLM providers, prompts, and test cases,
    and checks if the "Load Default Configuration" button works as expected.
    """
    at = AppTest.from_file("app.py").run()
    at.sidebar.selectbox[0].set_value("Define Prompts & Test Cases").run()

    # Verify initial display of data from MockSource (which are the defaults)
    assert at.json[0].value == DUMMY_LLM_PROVIDERS
    assert at.json[1].value == DUMMY_LLM_PROMPTS
    assert at.json[2].value == DUMMY_TEST_CASES

    # Click "Load Default Configuration" button
    at.button[0].click().run()
    assert at.success[0].value == "Default LLM Providers, Prompts, and Test Cases loaded!"
    assert at.session_state["llm_providers"] == DUMMY_LLM_PROVIDERS
    assert at.session_state["llm_prompts"] == DUMMY_LLM_PROMPTS
    assert at.session_state["test_cases"] == DUMMY_TEST_CASES


def test_run_evaluation_page():
    """
    Tests the "Run Evaluation" page.
    Simulates running the initial evaluation and verifies that results are populated
    in the session state and displayed in a DataFrame.
    """
    at = AppTest.from_file("app.py")
    # Set initial session state to simulate being on the page and having default config loaded
    at.session_state["current_page"] = "Run Evaluation"
    at.session_state["llm_providers"] = DUMMY_LLM_PROVIDERS
    at.session_state["llm_prompts"] = DUMMY_LLM_PROMPTS
    at.session_state["test_cases"] = DUMMY_TEST_CASES
    at.run()

    # Check initial state (info message)
    assert "Run the initial evaluation to see results." in at.info[0].value
    assert at.session_state["evaluation_results_df"] is None

    # Click the "Run Initial Evaluation" button
    at.button[0].click().run()

    # Verify that the evaluation ran and results are populated
    assert at.success[0].value == "Evaluation complete!"
    assert at.session_state["evaluation_results_df"] is not None
    assert isinstance(at.session_state["evaluation_results_df"], pd.DataFrame)
    assert not at.session_state["evaluation_results_df"].empty

    # Verify the dataframe is displayed
    pd.testing.assert_frame_equal(at.dataframe[0].value, DUMMY_EVALUATION_RESULTS_DF)


def test_analyze_results_page():
    """
    Tests the "Analyze Results" page.
    Pre-populates evaluation results and verifies that performance metrics,
    pass rates by provider/prompt, and failure patterns are correctly displayed.
    """
    at = AppTest.from_file("app.py")
    # Pre-populate session state with evaluation results to simulate a previous run
    at.session_state["current_page"] = "Analyze Results"
    at.session_state["evaluation_results_df"] = DUMMY_EVALUATION_RESULTS_DF
    at.run()

    # Verify overall performance metrics
    assert "Overall Performance Metrics" in at.subheader[0].value
    assert "Overall Pass Rate Across All Tests and Providers" in at.metric[0].label
    expected_overall_pass_rate = DUMMY_EVALUATION_RESULTS_DF["all_assertions_passed"].mean() * 100
    assert f"{expected_overall_pass_rate:.2f}%" in at.metric[0].value

    # Verify pass rate by provider
    assert "Pass Rate by Provider" in at.subheader[1].value
    pass_rate_by_provider = DUMMY_EVALUATION_RESULTS_DF.groupby("provider_id")["all_assertions_passed"].mean() * 100
    assert at.dataframe[0].value.iloc[0, 0] == f"{pass_rate_by_provider['openai']:.2f}%"
    assert at.dataframe[0].value.iloc[1, 0] == f"{pass_rate_by_provider['anthropic']:.2f}%"

    # Verify pass rate by prompt ID
    assert "Pass Rate by Prompt ID" in at.subheader[2].value
    pass_rate_by_prompt = DUMMY_EVALUATION_RESULTS_DF.groupby("prompt_id")["all_assertions_passed"].mean() * 100
    assert at.dataframe[1].value.iloc[0, 0] == f"{pass_rate_by_prompt['evidence_extraction']:.2f}%"

    # Verify visualizations are present (checking for subheader that precedes them)
    assert at.subheader[3].value == "Visualizations"

    # Verify analysis of failure patterns
    assert "Analysis of Failure Patterns" in at.subheader[4].value
    failed_tests_df_count = len(DUMMY_EVALUATION_RESULTS_DF[~DUMMY_EVALUATION_RESULTS_DF["all_assertions_passed"]])
    assert f"Total Failed Test Runs: {failed_tests_df_count}" in at.markdown[3].value
    if failed_tests_df_count > 0:
        assert "Top Failing Assertion Types" in at.markdown[4].value
        assert at.dataframe[2].value.iloc[0] == 1

        assert "Examples of Failed Test Runs" in at.markdown[6].value
        assert at.expander[0].label == "Failed Test: Handle ambiguous evidence (anthropic)"
        assert 'Failed Assertions' in at.expander[0].json[0].value
        assert at.expander[0].json[0].value['Failed Assertions'][0]['type'] == 'javascript'
    else:
        assert at.success[0].value == "No test runs failed in this initial run. Excellent work!"


def test_iterate_and_refine_page():
    """
    Tests the "Iterate & Refine" page.
    Simulates prompt refinement and re-evaluation, verifying that session state is updated
    and refined results are displayed, including the overall pass rate comparison.
    """
    at = AppTest.from_file("app.py")
    # Pre-populate session state with initial evaluation results
    at.session_state["current_page"] = "Iterate & Refine"
    at.session_state["evaluation_results_df"] = DUMMY_EVALUATION_RESULTS_DF
    at.run()

    # Click the "Simulate Prompt Refinement and Re-evaluate" button
    at.button[0].click().run()

    # Verify refined session state variables
    assert at.success[0].value == "Re-evaluation with refined prompt complete!"
    assert at.session_state["refined_llm_prompts"] is not None
    assert "evidence_extraction_v2" in at.session_state["refined_llm_prompts"]
    assert at.session_state["refined_test_cases"] is not None
    ambiguous_refined_test = next((t for t in at.session_state["refined_test_cases"] if t["description"] == "Handle ambiguous evidence"), None)
    assert ambiguous_refined_test is not None
    assert ambiguous_refined_test["prompt_id"] == "evidence_extraction_v2"

    assert at.session_state["re_evaluation_results_df"] is not None
    assert isinstance(at.session_state["re_evaluation_results_df"], pd.DataFrame)
    assert not at.session_state["re_evaluation_results_df"].empty

    # Verify refined results display for the specific test
    assert at.subheader[1].value == "Results for 'Handle ambiguous evidence' with refined prompt:"
    expected_refined_ambiguous_results = DUMMY_RE_EVALUATION_RESULTS_DF[
        (DUMMY_RE_EVALUATION_RESULTS_DF["description"] == "Handle ambiguous evidence") &
        (DUMMY_RE_EVALUATION_RESULTS_DF["prompt_id"] == "evidence_extraction_v2")
    ][["provider_id", "all_assertions_passed", "failed_assertions"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(at.dataframe[0].value, expected_refined_ambiguous_results)

    # Verify overall pass rate comparison
    assert at.subheader[3].value == "Overall Pass Rate Comparison"
    initial_pass_rate = DUMMY_EVALUATION_RESULTS_DF["all_assertions_passed"].mean() * 100
    refined_pass_rate = DUMMY_RE_EVALUATION_RESULTS_DF["all_assertions_passed"].mean() * 100
    assert at.markdown[2].value == f"Overall pass rate increased from {initial_pass_rate:.2f}% to {refined_pass_rate:.2f}% due to prompt refinement."


def test_final_report_and_regression_page():
    """
    Tests the "Final Report & Regression" page.
    Pre-populates both initial and re-evaluation results, verifies the pass rate
    comparison across versions and the generation of the evaluation report.
    """
    at = AppTest.from_file("app.py")
    # Pre-populate session state with both initial and re-evaluation results
    at.session_state["current_page"] = "Final Report & Regression"
    at.session_state["evaluation_results_df"] = DUMMY_EVALUATION_RESULTS_DF
    at.session_state["re_evaluation_results_df"] = DUMMY_RE_EVALUATION_RESULTS_DF

    # Simulate the refined_llm_prompts being set from the previous "Iterate & Refine" step
    at.session_state["refined_llm_prompts"] = DUMMY_LLM_PROMPTS.copy()
    at.session_state["refined_llm_prompts"]["evidence_extraction_v2"] = {
        "id": "evidence_extraction_v2",
        "raw": "Mock refined prompt raw content."
    }
    at.run()

    # Verify pass rate comparison across versions
    assert at.subheader[0].value == "Pass Rate Comparison Across Evaluation Versions"
    initial_pass_rates = DUMMY_EVALUATION_RESULTS_DF.groupby('provider_id')['all_assertions_passed'].mean() * 100
    refined_pass_rates = DUMMY_RE_EVALUATION_RESULTS_DF.groupby('provider_id')['all_assertions_passed'].mean() * 100

    # The displayed dataframe `at.dataframe[0].value` will be the unstacked comparison DataFrame
    assert at.dataframe[0].value.loc['Initial', 'openai'] == f"{initial_pass_rates['openai']:.2f}%"
    assert at.dataframe[0].value.loc['Initial', 'anthropic'] == f"{initial_pass_rates['anthropic']:.2f}%"
    assert at.dataframe[0].value.loc['Refined Prompt', 'openai'] == f"{refined_pass_rates['openai']:.2f}%"
    assert at.dataframe[0].value.loc['Refined Prompt', 'anthropic'] == f"{refined_pass_rates['anthropic']:.2f}%"

    # Verify the evaluation report is generated
    assert at.markdown[1].value.startswith("## LLM Evaluation Report")
    expected_initial_rate = DUMMY_EVALUATION_RESULTS_DF['all_assertions_passed'].mean() * 100
    expected_refined_rate = DUMMY_RE_EVALUATION_RESULTS_DF['all_assertions_passed'].mean() * 100
    expected_report_snippet = (
        f"Initial evaluation pass rate: {expected_initial_rate:.2f}%\n"
        f"Refined evaluation pass rate: {expected_refined_rate:.2f}%"
    )
    assert expected_report_snippet in at.markdown[1].value
