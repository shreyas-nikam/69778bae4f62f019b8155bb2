id: 69778bae4f62f019b8155bb2_user_guide
summary: Lab 14: Production Hardening & Governance User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: LLM Quality Assurance for AI Readiness

## 1. Introduction: Ensuring AI Readiness with Robust LLM Evaluation
Duration: 0:05:00

As a **Software Developer** at **PE Org-AI-R**, my team is building a critical AI-powered feature: an AI-readiness extractor. This tool analyzes corporate documents to identify evidence of a company's readiness across various AI dimensions like `data_infrastructure`, `ai_governance`, and `technology_stack`. Before we deploy this feature or update its underlying Large Language Models (LLMs), I need to guarantee that its outputs are consistently high-quality, accurate, and safe. Traditional manual testing is slow and doesn't scale, making automated evaluation indispensable.

This application outlines a real-world workflow to implement an automated LLM evaluation suite, conceptually similar to `promptfoo`, to systematically assess our LLM's performance. My goal is to rapidly identify issues, compare different LLM configurations, and ultimately ship features with confidence.

<aside class="positive">
<b>Why is this important?</b> Automated evaluation is crucial for maintaining the quality and reliability of AI-powered systems. It allows for rapid iteration, reduces the risk of deploying faulty features, and provides objective data to compare different LLMs or prompt strategies. This process ensures that our AI solutions meet high standards of accuracy and safety, which is vital for critical business applications like those at PE Org-AI-R.
</aside>

### Learning Objectives

In this lab, you will:
*   Understand the workflow of an automated LLM evaluation suite.
*   Define and utilize diverse test cases, including input prompts and validation logic (assertions).
*   Systematically test LLM responses against predefined quality criteria.
*   Analyze evaluation results to identify areas for improvement in prompts or model configurations.
*   Understand how to support testing across different LLM providers or prompt variations.

### Setting Up the Environment and Dependencies

The necessary libraries for our LLM evaluation workflow are pre-configured in this application. We'll simulate `promptfoo`'s functionality to directly integrate with our existing PE Org-AI-R systems. No manual setup is required on your part to get started with the evaluation process.

## 2. Define Prompts & Test Cases
Duration: 0:10:00

To begin our evaluation, we first need to define the different LLM providers and models we're considering, along with the core prompts our AI-readiness extractor will use. For PE Org-AI-R, we're currently exploring `OpenAI`'s `gpt-4o` and `Anthropic`'s `claude-sonnet`. Each model will have specific configuration parameters, such as `temperature`, which influences the creativity and randomness of the LLM's output.

This section of the application visually simulates the `providers` and `prompts` sections of a `promptfoo/config.yaml` file, showing you the structure of these definitions.

**Action:** Navigate to the "Define Prompts & Test Cases" section using the sidebar.

### Current LLM Providers

Observe the defined LLM providers. Each provider entry specifies an `id`, `label` (for display), and `model` (the specific LLM being used), along with `config` parameters like `temperature`.

```console
{
    "openai-gpt4o": {
        "id": "openai-gpt4o",
        "label": "OpenAI GPT-4o",
        "model": "gpt-4o",
        "config": {
            "temperature": 0.0
        }
    },
    "anthropic-claude-sonnet": {
        "id": "anthropic-claude-sonnet",
        "label": "Anthropic Claude Sonnet",
        "model": "claude-3-sonnet-20240229",
        "config": {
            "temperature": 0.1
        }
    }
}
```

### Current LLM Prompts

Next, examine the predefined LLM prompts. Each prompt has an `id` and a `raw` field containing the prompt template. Notice how prompts can include placeholders like `{{filing_excerpt}}`, which will be filled dynamically during evaluation.

```console
{
    "evidence_extraction": {
        "id": "evidence_extraction",
        "raw": "Extract AI-readiness evidence from this SEC filing excerpt. Ensure the output format is JSON.\n\nFiling: {{filing_excerpt}}\n\nReturn JSON with:\n- dimension: one of [data_infrastructure, ai_governance, technology_stack, talent, leadership, use_case_portfolio, culture]\n- evidence: the specific text or 'N/A' if no evidence\n- confidence: 0-1 score\n- reasoning: why this is evidence"
    }
}
```

### Crafting Diverse Test Cases and Assertions

A robust evaluation starts with diverse test cases that cover critical scenarios. As a Software Developer, I'm responsible for defining these tests, including the input variables (`vars`) and the `assert` conditions that define a "pass." These assertions act as our ground truth, verifying the LLM's output against expected formats, content, and quality metrics.

This section simulates the `tests` section of a `promptfoo/config.yaml`.

**Action:** Click the "Load Default Configuration" button if you haven't already. This ensures all necessary providers, prompts, and test cases are loaded into the application's memory for evaluation.

### Current Test Cases

Observe the list of test cases.

```console
[
    {
        "description": "Extract clear evidence of data infrastructure",
        "prompt_id": "evidence_extraction",
        "vars": {
            "filing_excerpt": "Our company has invested significantly in cloud-based data warehouses and advanced data lakes to support AI initiatives."
        },
        "assert": [
            { "type": "json" },
            { "type": "contains", "value": "data_infrastructure" },
            { "type": "contains", "value": "data warehouses" },
            { "type": "javascript", "value": "output.confidence >= 0.8" }
        ]
    },
    ... (more test cases)
]
```

### Explanation of Test Case Design

Each test case consists of:
*   A `description` for human readability, explaining the scenario.
*   A `prompt_id` linking to one of our predefined LLM prompts (e.g., `evidence_extraction`).
*   `vars`: A dictionary of input variables that will populate the prompt template, making the prompt dynamic. For instance, `{{filing_excerpt}}` in the prompt will be replaced by the value provided here.
*   `assert`: A list of conditions that the LLM's output must satisfy. These assertions are critical for defining what a "good" output looks like. They can check for:
    *   `type: json`: Ensures the output is valid JSON.
    *   `type: contains`: Verifies if the output contains specific text.
    *   `type: javascript`: Allows for complex logical checks on the parsed JSON output. For example, `output.confidence >= 0.8` checks if a numerical value meets a certain threshold.

For example, for an evidence extraction task, we might expect a confidence score $C$ for the extracted evidence to be above a certain threshold $T$, represented mathematically as $$ C \ge T $$
where $C$ is the confidence score (0-1) provided by the LLM, and $T$ is the minimum acceptable threshold (e.g., 0.7). This helps ensure the reliability of the extracted information.

## 3. Run Evaluation
Duration: 0:03:00

Now that our test cases and assertion logic are defined, I can run the automated evaluation suite. This involves iterating through each test case, dynamically rendering the prompt with the specified variables, "calling" each LLM provider, and then applying all defined assertions to the LLM's response. This systematic approach allows us to quickly assess how different models perform across various scenarios.

For this lab, we will simulate LLM responses to ensure reproducibility and avoid external API calls. The `mock_llm_response` function will generate plausible, structured responses based on the input prompt and test case.

**Action:** Navigate to the "Run Evaluation" section using the sidebar. Click the "Run Initial Evaluation" button. Wait for the spinner to disappear and the "Evaluation complete!" message to appear.

### Evaluation Results

Once the evaluation is complete, a table displaying the `Evaluation Results` will appear.

<aside class="positive">
The `evaluation_results_df` DataFrame now contains the results for each test case run against each LLM provider. Each row represents a specific `test_id` (input scenario) and `provider_id` (model). The `all_assertions_passed` column immediately tells me if the LLM output met all the predefined quality criteria for that specific test. The `failed_assertions` column provides details on why a test might have failed, including the type of assertion and the specific message. This granular data is crucial for debugging and understanding performance differences between models and prompt versions. It allows me to pinpoint exactly where the LLM's output deviates from our expectations.
</aside>

## 4. Analyze Results
Duration: 0:07:00

After running the evaluation, I need to analyze the results to understand the LLM's overall performance. This involves looking at aggregated metrics like pass/fail rates and drilling down into specific failure patterns. This step helps me quickly identify which LLMs are performing best and which areas of our AI-readiness extractor (e.g., specific dimensions or types of evidence) need improvement. Visualizations are key here to quickly grasp the larger picture.

**Action:** Navigate to the "Analyze Results" section using the sidebar.

### Overall Performance Metrics

You will see key metrics at the top:
*   **Overall Pass Rate Across All Tests and Providers**: This gives a high-level view of how well the system performs across all defined scenarios and models.
*   **Pass Rate by Provider**: Shows which LLM provider (e.g., OpenAI GPT-4o, Anthropic Claude Sonnet) performs better on average.
*   **Pass Rate by Prompt ID**: Indicates how well specific prompt types (e.g., `evidence_extraction`) are performing.

### Visualizations

Two bar charts provide a clear visual comparison:
*   **Pass Rate by LLM Provider**: Helps you quickly identify the top-performing models.
*   **Pass Rate by Prompt Type**: Shows which extraction tasks are more challenging or better handled by the current prompts.

### Analysis of Failure Patterns

This section details specific failures:
*   **Total Failed Test Runs**: An absolute count of runs that did not meet all assertions.
*   **Top Failing Assertion Types**: A breakdown (and a pie chart) showing which types of assertions (e.g., `json`, `contains`, `javascript`) are failing most frequently. This is crucial for understanding the root cause of issues.
*   **Examples of Failed Test Runs**: Provides snippets of actual LLM output and the associated failed assertions, allowing for detailed debugging.

### Explanation of Execution

The visualizations clearly show the `Pass Rate by LLM Provider` and `Pass Rate by Prompt Type`. This helps me understand which models are more reliable and which types of tasks (e.g., `evidence_extraction` vs. `dimension_scoring`) are more challenging for our current prompts and models. The `Distribution of Failing Assertion Types` pie chart pinpoints common failure modes. For instance, if `javascript` assertions (which often check numerical conditions like confidence scores or extracted values) are failing more often than `json` structure checks, it signals a specific type of problem.

For example, if `output.confidence >= 0.7` is frequently failing, it tells me the LLM is not consistently providing high-confidence evidence, or our threshold $T=0.7$ for acceptable confidence is too strict for the model's capabilities in certain scenarios. Adjusting the prompt to encourage higher confidence outputs or re-evaluating the threshold $T$ could be next steps.

The pass rate $P$ is calculated as:
$$ P = \frac{{\text{{Number of Passed Tests}}}}{{\text{{Total Number of Tests}}}} \times 100\% $$
where 'Number of Passed Tests' is the count of test runs where `all_assertions_passed` is `True`, and 'Total Number of Tests' is the total number of evaluation runs. This metric provides a high-level overview of the system's performance for easy comparison.

## 5. Iterate & Refine
Duration: 0:07:00

Identifying failures is the first step; the next is to iterate and improve. As a Software Developer, I'll use the failure analysis to refine our prompts and possibly adjust assertion thresholds. For instance, if the "Handle ambiguous evidence" test consistently fails because the confidence isn't low enough, I might need to clarify the prompt's instructions for uncertain scenarios or adjust the confidence threshold $T$.

Let's simulate a prompt refinement and re-evaluation for a specific failing test. We noticed our "Handle ambiguous evidence" test might be tricky. The original assertion was `output.confidence < 0.5 || output.dimension === null`. We could refine the prompt to explicitly instruct the LLM to return `null` for the dimension and a very low confidence if evidence is ambiguous, thereby improving its adherence to our defined quality criteria.

**Action:** Navigate to the "Iterate & Refine" section using the sidebar. Click the "Simulate Prompt Refinement and Re-evaluate" button. This will modify a prompt internally, apply it to a specific test case, and then run the evaluation again.

### Results for 'Handle ambiguous evidence' with refined prompt

You will see a comparison of the results for the specific test case "Handle ambiguous evidence" using both the original and refined prompts. The refined prompt should ideally show an improvement in `all_assertions_passed`.

### Overall Pass Rate Comparison

A bar chart compares the overall pass rates between the "Initial Evaluation" and the "Refined Prompt Evaluation". This visual comparison quickly shows the impact of your prompt engineering efforts on the overall system performance.

<aside class="positive">
This iterative process of analysis, refinement, and re-evaluation is the core of effective prompt engineering and LLM quality assurance. By addressing specific failure modes and validating changes through automated tests, we can continuously improve the performance and reliability of our AI features. You should see an overall pass rate increase due to prompt refinement.
</aside>

## 6. Final Report & Regression
Duration: 0:08:00

To prevent regressions with future deployments, it's crucial to track evaluation scores over time or across different model or prompt versions. This allows us to ensure that any new changes maintain or improve quality consistently. Finally, I'll generate a comprehensive 'LLM Evaluation Report' to summarize our findings for stakeholders, providing objective data for decision-making.

**Action:** Navigate to the "Final Report & Regression" section using the sidebar.

### Pass Rate Comparison Across Evaluation Versions

You will see a table and a bar chart comparing the pass rates by provider for both the "Initial" and "Refined Prompt" evaluations. This is a critical step for regression analysis.

### Generating the LLM Evaluation Report

A detailed markdown report summarizing the evaluation findings will be generated and displayed.

### Explanation of Results and Report

The "Pass Rate Comparison Across Evaluation Versions" visualization is a clear example of **Regression Analysis**. It shows how the performance of each LLM provider changed from the 'Initial' evaluation to the 'Refined Prompt' evaluation. Ideally, all bars in the 'Refined Prompt' section should be equal to or higher than their 'Initial' counterparts, indicating no regressions and overall improvement. This objective, quantitative comparison is vital for validating changes.

The `LLM Evaluation Report` provides a structured, detailed summary for stakeholders. It covers the evaluation setup, comparative performance metrics across different LLM providers and prompt types, key findings from the analysis, and actionable recommendations for future development. This report is our deliverable, allowing the team to make informed decisions about LLM selection, prompt strategies, and deployment readiness, backed by objective data. It emphasizes the importance of automated evaluation for maintaining and continuously improving the quality of our AI-powered features, ensuring they meet the high standards expected by PE Org-AI-R.
