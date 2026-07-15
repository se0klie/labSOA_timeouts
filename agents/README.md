# RPC timeout eligibility classifier

This is Agent 1 only: it identifies relevant server-side/internal RPC timeout issues and captures exact supporting excerpts

Install the official Gemini SDK (Python 3.10+):

```bash
python -m pip install -U google-genai
export GEMINI_API_KEY='your-key'
```

Run it:

```bash
python classifier_agent.py --input whole_jira_csv.csv --output classified_issues.csv --model gemini-3.1-flash-lite-preview --max-rows 100
```

```bash
python heuristics_agent.py --input whole_jira_csv.csv --classifications classified_issues.csv --output heuristics.csv --model gemini-3.1-flash-lite-preview --max-rows 100
```



