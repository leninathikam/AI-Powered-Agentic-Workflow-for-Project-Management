"""AI-Powered Agentic Workflow — recruiter-ready Streamlit demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from email_router_workflow.openai_config import (  # noqa: E402
    format_openai_config_error,
    resolve_openai_base_url,
)
from email_router_workflow.workflow import (  # noqa: E402
    DEFAULT_WORKFLOW_PROMPT,
    load_product_spec,
    run_workflow,
)

st.set_page_config(
    page_title="Agentic Workflow | Project Planning",
    page_icon="🗂️",
    layout="wide",
)

SECTION_TITLES = {
    "stories": "User Stories",
    "features": "Product Features",
    "tasks": "Engineering Tasks",
    "general": "General Output",
}


def _secret_get(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def ensure_openai_config() -> str | None:
    """Load API key + optional base URL from Streamlit secrets, else optional dotenv/env."""
    key = _secret_get("OPENAI_API_KEY") or _secret_get("UDACITY_OPENAI_API_KEY")
    base_url = _secret_get("OPENAI_BASE_URL")

    if not key:
        try:
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
            load_dotenv()
        except ImportError:
            pass
        key = os.getenv("OPENAI_API_KEY") or os.getenv("UDACITY_OPENAI_API_KEY")
        if key:
            key = key.strip()
        if not base_url:
            env_base = (os.getenv("OPENAI_BASE_URL") or "").strip()
            base_url = env_base or None

    if key:
        os.environ["OPENAI_API_KEY"] = key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    elif key and not (os.getenv("OPENAI_BASE_URL") or "").strip():
        os.environ["OPENAI_BASE_URL"] = resolve_openai_base_url(key)

    return key


def main():
    st.title("AI-Powered Agentic Workflow")
    st.caption(
        "Multi-agent project planning: product brief → user stories → features → engineering tasks"
    )

    api_key = ensure_openai_config()
    base_url = resolve_openai_base_url(api_key) if api_key else None

    with st.sidebar:
        st.header("About")
        st.write(
            "This demo runs an agentic planning workflow for an Email Router product. "
            "A planning agent extracts steps, then specialist agents (with evaluation loops) "
            "produce structured deliverables."
        )
        st.markdown(
            "[GitHub repo](https://github.com/leninathikam/AI-Powered-Agentic-Workflow-for-Project-Management)"
        )
        st.divider()
        st.write("API key:", "ready" if api_key else "missing")
        if base_url:
            st.write("API base URL:", base_url)
        st.caption(
            "sk-... keys use api.openai.com; voc-... keys use openai.vocareum.com. "
            "Override with OPENAI_BASE_URL if needed."
        )
        st.subheader("Pipeline")
        st.markdown(
            "1. Action planning\n"
            "2. Product Manager → user stories\n"
            "3. Program Manager → features\n"
            "4. Tech Lead → engineering tasks"
        )

    try:
        product_spec = load_product_spec()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load product spec: {exc}")
        st.stop()

    col_input, col_output = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("Product brief")
        with st.expander("Email Router specification", expanded=True):
            st.text(product_spec[:3500] + ("..." if len(product_spec) > 3500 else ""))

        prompt = st.text_area(
            "Workflow prompt",
            value=DEFAULT_WORKFLOW_PROMPT,
            height=120,
        )
        run = st.button("Run workflow", type="primary", use_container_width=True)

    with col_output:
        st.subheader("Generated plan")
        if not run:
            st.info("Click **Run workflow** to generate stories, features, and tasks.")
            return

        if not api_key:
            st.error(
                "Add `OPENAI_API_KEY` in Streamlit secrets or a local `.env` file. "
                "Use `sk-...` for OpenAI or `voc-...` for Vocareum. "
                "Optional: `OPENAI_BASE_URL` to override the endpoint."
            )
            return

        with st.spinner("Agents are planning, routing, and evaluating..."):
            try:
                result = run_workflow(prompt)
            except Exception as exc:  # noqa: BLE001
                st.error(format_openai_config_error(exc))
                return

        steps = result.get("steps", [])
        if not steps:
            st.warning("Workflow finished with no steps.")
            return

        tabs = st.tabs([SECTION_TITLES.get(s["type"], s["type"].title()) for s in steps])
        for tab, step in zip(tabs, steps):
            with tab:
                st.markdown(step["output"])

        with st.expander("Raw workflow result"):
            st.json(result)


if __name__ == "__main__":
    main()
