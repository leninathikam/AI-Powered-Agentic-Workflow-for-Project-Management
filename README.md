# Email Router Workflow

An agentic workflow for turning an Email Router product brief into structured user stories, product features, and engineering tasks.

## Streamlit demo (recruiter showcase)

### Local

```bash
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY
streamlit run streamlit_app.py
```

### Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with **Main file path**: `streamlit_app.py`
4. In **Secrets**, add either a standard OpenAI key or a Vocareum key:

```toml
# OpenAI (auto uses https://api.openai.com/v1)
OPENAI_API_KEY = "sk-..."

# Or Vocareum (auto uses https://openai.vocareum.com/v1)
# OPENAI_API_KEY = "voc-..."

# Optional override if auto-detection is wrong:
# OPENAI_BASE_URL = "https://api.openai.com/v1"
```

Also works on **Render** with start command:
`streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

## Layout

```text
src/email_router_workflow/   Core package with reusable agents and the workflow
scripts/                     CLI entrypoint for running the workflow
data/product_specs/          Product specification input used by the workflow
artifacts/                   Output and runtime scratch space
streamlit_app.py             Interactive demo UI
```

## Setup

```bash
pip install -r requirements.txt
```

or editable:

```bash
pip install -e .
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (`sk-...` or `voc-...`).
`OPENAI_BASE_URL` is optional — it is inferred from the key prefix when omitted.

## CLI run

```bash
python scripts/run_workflow.py
```

The workflow reads the Email Router spec from `data/product_specs/email_router.txt`, runs the planning and routing agents, and prints the final structured plan to the console.
