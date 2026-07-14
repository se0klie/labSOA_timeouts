# RPC timeout eligibility classifier

This is Agent 1 only: it identifies relevant server-side/internal RPC timeout issues and captures exact supporting excerpts

Install the official Gemini SDK (Python 3.10+):

```bash
python -m pip install -U google-genai
export GEMINI_API_KEY='your-key'
```

Run it:

```bash
python classify_issues.py --input selected_issues.csv --output classified_issues.csv --model gemini-3.5-flash --max-rows 100
```
