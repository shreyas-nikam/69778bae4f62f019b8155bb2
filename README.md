# QuLab: Lab 14: Production Hardening & Governance - LLM Quality Assurance with Automated Evaluation

<p align="center">
  <img src="https://www.quantuniversity.com/assets/img/logo5.jpg" alt="QuantUniversity Logo" width="150"/>
</p>

## 🛡️ LLM Quality Assurance with Automated Evaluation for PE Org-AI-R

This Streamlit application is a hands-on lab project designed to simulate a robust LLM evaluation workflow, conceptually similar to `promptfoo`, for the **PE Org-AI-R** company. As a Software Developer, you'll learn to systematically assess Large Language Model (LLM) performance for critical AI-powered features, ensuring high-quality, accurate, and safe outputs before deployment or updates. The focus is on automating the evaluation process to rapidly identify issues, compare configurations, and ship features with confidence.

### Learning Objectives

In this lab, you will:

*   Implement an automated LLM evaluation suite using Python to simulate core `promptfoo` functionalities.
*   Define diverse test cases, including input prompts and validation logic (assertions).
*   Systematically test LLM responses against predefined quality criteria.
*   Analyze evaluation results to identify areas for improvement in prompts or model configurations.
*   Understand how to support testing across different LLM providers or prompt variations.
*   Perform iterative refinement and regression analysis.

---

## ✨ Features

This application guides you through a comprehensive LLM evaluation lifecycle, featuring:

*   **Interactive Streamlit UI**: A user-friendly interface for navigating the evaluation process.
*   **LLM Provider & Prompt Definition**: Configure and view simulated LLM providers (e.g., `OpenAI`, `Anthropic`) and core prompt templates used by the AI-readiness extractor.
*   **Test Case Management**: Define rich test cases with input variables (`vars`) and powerful assertion logic (`assert` types like `json`, `contains`, `javascript`) to validate LLM outputs against expected criteria.
*   **Automated Evaluation Suite**: Run a simulated evaluation process that iterates through test cases, "calls" LLMs (using mock responses for reproducibility), and applies assertions.
*   **Detailed Results Analysis**: View raw evaluation results in a structured DataFrame, including pass/fail status and detailed failure messages.
*   **Aggregated Performance Metrics**: Instantly see overall pass rates, and breakdown by LLM provider and prompt ID.
*   **Visualizations**: Bar charts and pie charts to quickly grasp performance differences and common failure patterns.
*   **Iterative Refinement Simulation**: Experience how prompt engineering can address identified weaknesses by modifying a prompt and re-evaluating.
*   **Regression Analysis**: Compare performance across different evaluation versions (initial vs. refined) to track improvements and prevent regressions.
*   **Comprehensive Evaluation Report**: Generate a summary report for stakeholders, outlining findings, metrics, and recommendations.

---

## 🚀 Getting Started

Follow these instructions to set up and run the Streamlit application locally.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (if this project is hosted on a Git repository):
    ```bash
    git clone <repository_url>
    cd QuLab-Lab14-LLM-Evaluation
    ```
    *(Assuming a project folder name `QuLab-Lab14-LLM-Evaluation`)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in the project root with the following content:
    ```
    streamlit
    pandas
    matplotlib
    seaborn
    pyyaml
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure `source.py` is present**:
    Make sure the `source.py` file (containing the core logic for LLM providers, prompts, test cases, evaluation functions, and mock responses) is in the same directory as `app.py`.

### Project Structure

```
.
├── app.py                     # Main Streamlit application
├── source.py                  # Contains LLM definitions, prompts, test cases, and evaluation logic
└── requirements.txt           # Python dependencies
```

---

## 🏃‍♀️ Usage

To run the application, execute the following command in your terminal from the project's root directory:

```bash
streamlit run app.py
```

This will open the Streamlit application in your default web browser.

### Application Walkthrough

The application is structured with a sidebar navigation to guide you through the LLM evaluation workflow:

1.  **Introduction & Setup**: Provides an overview of the lab and its objectives.
2.  **Define Prompts & Test Cases**:
    *   View the predefined LLM providers and prompts.
    *   Examine the diverse test cases and their assertion logic.
    *   **Action**: Click the "Load Default Configuration" button to initialize session state with the predefined settings.
3.  **Run Evaluation**:
    *   **Action**: Click "Run Initial Evaluation" to execute the simulated LLM evaluation suite.
    *   Review the detailed results DataFrame.
4.  **Analyze Results**:
    *   Examine aggregated performance metrics (overall pass rates, pass rates by provider/prompt).
    *   Review visualizations that highlight performance trends and common failure types.
    *   Analyze examples of failed test runs.
5.  **Iterate & Refine**:
    *   **Action**: Click "Simulate Prompt Refinement and Re-evaluate" to see how a refined prompt impacts specific test cases and overall performance.
    *   Compare the refined results against the initial run.
6.  **Final Report & Regression**:
    *   View a comparison of pass rates across evaluation versions to identify regressions or improvements.
    *   Read the automatically generated "LLM Evaluation Report" summarizing the entire process and findings.

---

## 🛠️ Technology Stack

*   **Python 3.8+**: The core programming language.
*   **Streamlit**: For creating interactive web applications with pure Python.
*   **Pandas**: For data manipulation and analysis of evaluation results.
*   **Matplotlib & Seaborn**: For creating insightful data visualizations.
*   **PyYAML**: For handling YAML configuration (simulating `promptfoo`'s config structure).
*   **`json`**, **`re`**, **`os`**, **`typing`**: Standard Python libraries for various utilities.

---

## 🙌 Contributing

This is a lab project, primarily for educational purposes. However, if you have suggestions for improvements or find issues, feel free to:

1.  Fork the repository (if applicable).
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(A `LICENSE` file would typically be included in a real project.)*

---

## 📞 Contact

For questions or feedback regarding this lab, please reach out to QuantUniversity.

*   **Website**: [QuantUniversity.com](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com

---
