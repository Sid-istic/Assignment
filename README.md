

# 🧩 **Kasparro Agentic Facebook Analyst – Assignment Submission**

This repository contains my implementation of the **Agentic Facebook Analyst System**, as specified in the assignment PDF.
The project follows a fully modular, multi-agent design where each agent performs one stage of the analysis pipeline (planning → data analysis → insight generation → evaluation → creative generation).

---

# 🚀 **1. Overview**

This project simulates an **agentic AI system** designed to perform:

* Facebook Ads data analysis
* ROAS & CTR diagnosis
* Hypothesis generation
* Insight evaluation with confidence scoring
* Creative improvement generation
* Complete agent-to-agent orchestration

The system is built using **Python**, following the exact architecture described in the assignment.
No LLM calls are required; rule-based agents are implemented so the project runs **without API keys**.

---

# 🧠 **2. System Architecture**

The system uses **five agents**, coordinated by an **Orchestrator**:

### **1. Planner Agent**

* Breaks down the user query
* Produces ordered tasks
* Defines dependencies between agents

### **2. Data Agent**

* Loads Facebook Ads dataset
* Validates required columns
* Produces global metrics, ROAS trends, CTR trends
* Identifies low-CTR ads

### **3. Insight Agent**

* Generates hypotheses based on the data
* Hypotheses follow assignment-required structure

### **4. Evaluator Agent**

* Validates each hypothesis
* Computes quantitative evidence (ROAS drop, CTR drop, spend share, etc.)
* Assigns a confidence score: **high / medium / low**

### **5. Creative Agent**

* Uses data from low-CTR ads
* Generates improved creative ideas, headlines, hooks, and CTAs
* Outputs marketer-ready suggestions

### **Orchestrator**

* Manages the full workflow end-to-end
* Logs JSON traces for each stage
* Writes final insights & creative recommendations
* Generates a final human-readable `report.md`

---

# 📂 **3. Repository Structure**

```
kasparro-agentic-fb-analyst/
│
├── run.py
├── README.md
├── agent_graph.md
├── requirements.txt
├── Makefile
├── run.sh
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── orchestrator/
│   │   └── orchestrator.py
│   ├── agents/
│   │   ├── planner.py
│   │   ├── data_agent.py
│   │   ├── insight_agent.py
│   │   ├── evaluator_agent.py
│   │   └── creative_agent.py
│   └── utils/
│       ├── data_loader.py
│       ├── logging_utils.py
│       └── metrics.py
│
├── prompts/
│   ├── planner_prompt.md
│   ├── data_agent_prompt.md
│   ├── insight_agent_prompt.md
│   ├── evaluator_prompt.md
│   └── creative_prompt.md
│
├── data/
│   ├── sample_fb_ads.csv
│   └── README.md
│
├── reports/
│   ├── report.md
│   ├── insights.json
│   └── creatives.json
│
└── tests/
    └── test_evaluator.py
```

---

# ⚙️ **4. Installation & Setup**

## **Create environment**

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux
```

## **Install dependencies**

```bash
pip install -r requirements.txt
```

## **Run the system**

```bash
python run.py "Analyze ROAS drop"
```

Outputs will appear in the `reports/` folder.

---

# 📊 **5. Input & Output Expectations**

### **Input**

A marketer-style query, such as:

> “Analyze ROAS drop and find likely causes.”

### **Output (Generated Automatically)**

#### 1. **insights.json**

Structured hypotheses with:

* Title
* Description
* Driver
* Reasoning
* Confidence
* Evidence

#### 2. **creatives.json**

New creative suggestions with:

* Headline
* Primary text
* CTA
* Based on original creative message

#### 3. **report.md**

A clean, marketer-friendly summary including:

* Account summary
* Hypotheses + confidence
* Creative improvement ideas

#### 4. **JSON Logs (Traces)**

Stored in `logs/`:

* planner trace
* data summary trace
* insight generation trace
* evaluation trace
* creative generation trace

(As required by assignment.)

---

# 📝 **6. Prompts (LLM Design Requirement)**

The `prompts/` directory contains:

* Planner Agent Prompt
* Data Agent Prompt
* Insight Agent Prompt
* Evaluator Agent Prompt
* Creative Agent Prompt

Each prompt includes:

* **Think → Analyze → Conclude** format
* **Strict JSON output** requirements
* Agent roles
* Instructions for deterministic output

These satisfy the prompt engineering requirements in the assignment.

---

# 🧪 **7. Tests**

A minimal unit test is included:

```
tests/test_evaluator.py
```

It validates the confidence scoring logic of the Evaluator Agent.

---

# 🧱 **8. Reproducibility**

The system uses:

* Seeded randomness
* Version-pinned `requirements.txt`
* Sample dataset (`sample_fb_ads.csv`)
* Config-driven thresholds
* Generated logs and reports committed for review

---

# 🏷️ **9. Versioning**

The repository includes:

* Multiple commits
* Clear commit messages
* A **v1.0 tag** as required
* A **self-review pull request** describing the system
  (as per assignment instructions)

---

# 📌 **10. How to Run for Review**

To reproduce the final output:

```bash
python run.py "Analyze ROAS drop"
```

The evaluator will find:

* ROAS degradation (if present)
* CTR decline
* Underperforming campaigns
* Low-CTR creatives needing refresh

Creatives and insights appear under `reports/`.

---

# 🎯 **11. Key Features Implemented (Matching PDF Requirements)**

* ✔ Multi-agent architecture
* ✔ Clear agent roles & boundaries
* ✔ Functional pipeline
* ✔ JSON traces for each stage
* ✔ Error handling
* ✔ Data validation
* ✔ Hypothesis generation
* ✔ Confidence scoring + evidence
* ✔ Creative variant generation
* ✔ Final report in markdown
* ✔ Git versioning & tagging
* ✔ Self-review PR
* ✔ No API keys required
* ✔ Deterministic output

---

# 🏁 **12. Final Notes**

This project is built to match the exact expectations of the assignment, including:

* Reproducibility
* Agent-to-agent orchestration
* Comprehensive reporting
* Well-documented code
* A clear, professional repository

If needed, I can also create:

✅ A professional self-review
✅ A GitHub PR description
✅ A demo video script

Just tell me!
