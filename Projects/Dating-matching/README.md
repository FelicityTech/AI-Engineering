# 🤖 Couple Matching Probability — AI-Powered Speed Dating Match Prediction Agent

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.8-purple?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-yellow?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-1.2.10-green?style=flat-square)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-OpenAI-black?style=flat-square&logo=openai)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📖 Overview

What makes two people click in just four minutes? Can machine learning predict romantic chemistry before a date even ends?

This project answers exactly that — by building an **autonomous AI agent** that takes a single natural language instruction and independently orchestrates an entire machine learning pipeline: from raw messy data all the way to a trained probability model with an **ROC AUC score of 0.8341**.

Unlike traditional ML pipelines where you manually chain each step, this project uses **LangGraph** and the **ReAct (Reason + Act) pattern** to create an intelligent agent that *reasons* about what needs to happen next and *acts* by calling the right tool at the right time — completely on its own.

---

## 🎯 Project Highlights

- ✅ Built an end-to-end **agentic ML workflow** from a single prompt
- ✅ Achieved **ROC AUC = 0.8341** on real speed dating data
- ✅ Automated data cleaning, feature selection, and model training with **zero manual intervention**
- ✅ Identified the top 10 most predictive features for romantic matching
- ✅ Handled real-world data challenges: byte-string encodings, missing values, and data leakage

---

## 🧠 What Is an Agentic Workflow?

Traditional ML pipelines require you to:
- Manually write each step in the correct order
- Pass data between functions yourself
- Track intermediate results
- Handle errors at every stage

An **agentic workflow** delegates all of this to an AI. You give it a goal — the agent figures out the path.

This project demonstrates that pattern end-to-end using **LangGraph's StateGraph** and the **ReAct loop**:

```
[Reason] → What should I do next?
[Act]    → Call the appropriate tool
[Observe]→ Process the tool's output
[Repeat] → Until the task is complete
```

---

## 📊 Dataset

**Source:** [Speed Dating Dataset — Kaggle (Ulrik Thyge Pedersen)](https://www.kaggle.com/datasets/ulrikthygepedersen/speed-dating/data)

| Property | Detail |
|---|---|
| Rows | 8,378 |
| Target Variable | `match` (1 = both said yes, 0 = otherwise) |
| Key Features | Attractiveness, sincerity, intelligence, fun, ambition, shared interests ratings |
| Data Challenges | Byte-string encodings, missing values, leakage columns |

### ⚠️ Data Leakage Addressed
The columns `decision` and `decision_o` (individual yes/no votes) were deliberately removed. Knowing both votes makes the match deterministic — that's not prediction, that's peeking. The model predicts purely from ratings and demographics.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              LangGraph StateGraph            │
│                                              │
│   ┌──────────┐      ┌──────────────────┐    │
│   │ Scientist│─────▶│   Tools Node     │    │
│   │  (LLM)   │◀─────│                  │    │
│   └──────────┘      │ • clean_data     │    │
│        │            │ • select_features│    │
│        ▼            │ • train_model    │    │
│      [END]          └──────────────────┘    │
└─────────────────────────────────────────────┘
```

**Router Logic:**
- If the LLM response contains tool calls → route to `tools` node
- If no tool calls remain → route to `END`
- After every tool execution → return to `scientist` node

---

## 🛠️ Tools Defined

### 1. `clean_speed_dating_data(file_name)`
Handles all preprocessing in one pass:
- Drops leakage columns (`decision`, `decision_o`, `has_null`)
- Strips byte-string encodings (e.g., `b'female'` → `female`)
- Label-encodes all categorical variables
- Imputes missing values using median strategy
- Saves cleaned output to `cleaned_data.csv`

### 2. `select_top_features(n_features)`
Runs intelligent feature selection:
- Uses **Recursive Feature Elimination (RFE)** backed by a Random Forest
- Ranks features by predictive importance
- Returns the top N features for model training

**Top 10 Features Selected:**
| # | Feature | What It Means |
|---|---|---|
| 1 | `attractive_o` | How attractive you rated your partner |
| 2 | `like` | Overall how much you liked them |
| 3 | `guess_prob_liked` | How likely you think they liked you |
| 4 | `interests_correlate` | Correlation of your interests |
| 5 | `shared_interests_o` | Shared interests score |
| 6 | `funny_o` | How funny you rated them |
| 7 | `attractive_partner` | How important attractiveness was to you |
| 8 | `attractive_important` | Attractiveness preference weight |
| 9 | `pref_o_attractive` | Partner's attractiveness preference |
| 10 | `field` | Field of study |

### 3. `train_probability_model(features)`
Trains and evaluates the final model:
- Splits data 80/20 with stratification
- Trains an **XGBoost classifier**
- Predicts match *probabilities* (not just binary labels)
- Evaluates using **ROC AUC** — the gold standard for probability ranking

---

## 📈 Results

| Metric | Score |
|---|---|
| **ROC AUC** | **0.8341** |
| Training Data | 6,702 rows |
| Test Data | 1,676 rows |
| Features Used | 10 |

> An ROC AUC of 0.8341 means the model correctly ranks a true match above a non-match **83.41% of the time** — using only ratings and demographics, with no knowledge of individual decisions.

---

## ⚙️ Tech Stack

| Tool | Version | Role |
|---|---|---|
| **LangGraph** | 1.0.8 | Agentic workflow orchestration |
| **LangChain** | 1.2.10 | LLM integration & tool binding |
| **GPT-4o-mini** | via OpenAI | Reasoning engine (ReAct brain) |
| **XGBoost** | 3.2.0 | Probability model training |
| **Scikit-learn** | 1.8.0 | RFE feature selection & preprocessing |
| **Pandas** | 3.0.0 | Data manipulation |
| **NumPy** | 2.4.2 | Numerical operations |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/couple-matching-probability.git
cd couple-matching-probability

# Install dependencies
pip install langchain==1.2.10 langchain-openai==1.1.9 langgraph==1.0.8 \
            openai==2.20.0 numpy==2.4.2 pandas==3.0.0 \
            scikit-learn==1.8.0 xgboost==3.2.0
```

### Set Your API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Download the Dataset

```bash
wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/sqiJ_CW9x_k2T6C2KgPf6Q/speeddating.csv
```

### Run the Agent

```python
from langchain_core.messages import HumanMessage

query = "Clean 'speeddating.csv', select top 10 features, and predict match probability."

for output in app.stream({"messages": [HumanMessage(content=query)]}, stream_mode="updates"):
    for node_name, state_update in output.items():
        if node_name == "scientist":
            message = state_update["messages"][-1]
            if message.tool_calls:
                print(f"🤔 THOUGHT: Calling {[t['name'] for t in message.tool_calls]}")
            else:
                print(f"✅ FINAL ANALYSIS:\n{message.content}")
        elif node_name == "tools":
            for tool_msg in state_update["messages"]:
                print(f"👁️ OBSERVATION: {str(tool_msg.content)[:300]}")
```

### Expected Output

```
🤔 THOUGHT: Calling ['clean_speed_dating_data']
👁️ OBSERVATION: Data cleaned. Rows: 8378. Saved to 'cleaned_data.csv'...

🤔 THOUGHT: Calling ['select_top_features']
👁️ OBSERVATION: {"selected_features": ["attractive_o", "like", "guess_prob_liked", ...]}

🤔 THOUGHT: Calling ['train_probability_model']
👁️ OBSERVATION: Model trained. ROC AUC Score: 0.8341. Predictions are reliable...

✅ FINAL ANALYSIS:
The model achieved a ROC AUC of 0.8341 using the top 10 features...

🏁 Workflow Complete.
```

---

## 🔬 Extending the Project

Some ideas to take this further:

- **Hyperparameter tuning** — Let the agent experiment with XGBoost parameters autonomously
- **Multi-model comparison** — Add tools for Logistic Regression, LightGBM, and let the agent pick the best
- **Feature count experiments** — Run the pipeline with 5, 10, and 15 features and compare AUC scores
- **SHAP explainability** — Add a tool that generates feature importance visualizations
- **Real-time prediction** — Build a simple UI where users input ratings and get a match probability score

---

## 📁 Project Structure

```
couple-matching-probability/
│
├── Couple_Matching_Probability.ipynb   # Main notebook
├── speeddating.csv                     # Raw dataset
├── cleaned_data.csv                    # Cleaned dataset (generated)
└── README.md                           # You are here
```

---

## 🙋 Author

**Solomon Eniola Adegoke**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/solomon-eniola-adegoke/)

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and build on it.

---

> *"Real connection — between people or between a model and the truth — comes down to the right signals, handled the right way."*
