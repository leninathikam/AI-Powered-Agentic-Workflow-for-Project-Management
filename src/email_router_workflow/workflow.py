from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

from .openai_config import resolve_openai_base_url
from .workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_SPEC_PATH = REPO_ROOT / "data" / "product_specs" / "email_router.txt"
DEFAULT_WORKFLOW_PROMPT = (
    "Create user stories, product features, and engineering tasks for the Email Router "
    "product described in the product spec."
)


def load_openai_api_key() -> str:
    """Load the OpenAI API key (and optional base URL) from environment variables.

    dotenv is optional — Streamlit Cloud typically injects secrets into the
    environment / st.secrets without a .env file.
    """
    load_dotenv()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("UDACITY_OPENAI_API_KEY")
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to Streamlit secrets, your environment, or a .env file."
        )
    # Ensure a base URL is available for clients; auto-pick from key prefix when unset.
    if not (os.getenv("OPENAI_BASE_URL") or "").strip():
        os.environ["OPENAI_BASE_URL"] = resolve_openai_base_url(api_key)
    return api_key


def load_product_spec() -> str:
    """Read the Email Router product specification from the data directory."""
    return PRODUCT_SPEC_PATH.read_text(encoding="utf-8")


def get_step_type(step: str) -> str:
    """Identify which deliverable type a workflow step should produce."""
    step_lower = step.lower()
    if "define user stories" in step_lower or step_lower.lstrip().startswith("1."):
        return "stories"
    if "define product features" in step_lower or step_lower.lstrip().startswith("2."):
        return "features"
    if "engineering tasks" in step_lower or step_lower.lstrip().startswith("3."):
        return "tasks"
    return "general"


def build_step_query(step: str, previous_output: str | None = None, user_stories: str | None = None) -> str:
    """Add product and prior-step context so each agent produces grounded deliverables."""
    step_type = get_step_type(step)
    query = (
        f"{step}\n\n"
        "Use the Email Router product specification provided in your knowledge. "
        "Return actual project deliverables, not instructions about how to write deliverables."
    )

    if step_type == "stories":
        query += (
            "\n\nRequired format for every story on its own line: "
            "As a [type of user], I want [an action or feature] so that [benefit/value]."
        )
    elif step_type == "features":
        query += (
            "\n\nRequired format for every feature:\n"
            "Feature Name: ...\nDescription: ...\nKey Functionality: ...\nUser Benefit: ...\n"
            "Produce exactly 5 features."
        )
        if user_stories:
            query += f"\n\nGroup these user stories into features:\n{user_stories}"
    elif step_type == "tasks":
        query += (
            "\n\nRequired format for every task:\n"
            "Task ID: ...\nTask Title: ...\nRelated User Story: ...\nDescription: ...\n"
            "Acceptance Criteria: ...\nEstimated Effort: ...\nDependencies: ..."
        )
        if user_stories:
            query += f"\n\nReference these user stories in Related User Story:\n{user_stories}"
        if previous_output:
            query += f"\n\nUse these product features as context:\n{previous_output}"

    return query


def route_step(step_type: str, step_query: str, routing_agent: RoutingAgent) -> str:
    """Route a workflow step to the correct specialist team."""
    agent_index = {"stories": 0, "features": 1, "tasks": 2}
    selected_agents = [routing_agent.agents[agent_index[step_type]]]
    step_router = RoutingAgent(routing_agent.openai_api_key, selected_agents)
    return step_router.route(step_query)


def build_workflow_agents(openai_api_key: str, product_spec: str) -> dict[str, Any]:
    """Create the reusable agents and routing functions used by the workflow."""
    knowledge_action_planning = (
        "Stories are defined from a product spec by identifying a "
        "persona, an action, and a desired outcome for each story. "
        "Each story represents a specific functionality of the product "
        "described in the specification. \n"
        "Features are defined by grouping related user stories. \n"
        "Tasks are defined for each story and represent the engineering "
        "work required to develop the product. \n"
        "A development plan for a product contains all these components. \n"
        "When asked to create a full project plan, always extract exactly these steps in order: "
        "1. Define user stories from the product spec. "
        "2. Define product features by grouping the user stories. "
        "3. Define engineering tasks for implementing the user stories."
    )
    action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

    persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
    knowledge_product_manager = (
        "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
        "The sentences always start with: As a "
        "Write at least 4 user stories for the Email Router product spec below. "
        "Each story must be on its own line and use this exact format: "
        "As a [type of user], I want [an action or feature] so that [benefit/value]. "
        "Do not use numbered lists or bullet points for user stories. "
        + product_spec
    )
    product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona_product_manager,
        knowledge_product_manager,
    )

    persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
    evaluation_criteria_product_manager = (
        "The answer must contain at least 4 Email Router user stories. "
        "Every story must follow this exact structure on its own line: "
        "As a [type of user], I want [an action or feature] so that [benefit/value]. "
        "Respond No if stories use numbered lists, bullet points, or headings instead of the required sentence format. "
        "Respond No if the answer contains Feature Name:, Task ID:, or any product feature or engineering task labels."
    )
    product_manager_evaluation_agent = EvaluationAgent(
        openai_api_key,
        persona_product_manager_eval,
        evaluation_criteria_product_manager,
        product_manager_knowledge_agent,
        max_interactions=10,
    )

    persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
    knowledge_program_manager = (
        "Features of a product are defined by organizing similar user stories into cohesive groups. "
        "Write exactly 5 Email Router product features based on the product spec below. "
        "Cover email ingestion, NLP classification, automated responses, routing to SMEs, "
        "dashboard monitoring, and knowledge base integration across the five features. "
        "Each feature MUST use these exact labels on separate lines:\n"
        "Feature Name: <title>\n"
        "Description: <brief explanation>\n"
        "Key Functionality: <specific capabilities>\n"
        "User Benefit: <value to the user>\n"
        "Separate each feature with a blank line. "
        "Do not use numbered lists, bullet points, or markdown headings. "
        "Features must relate to email ingestion, NLP classification, automated responses, "
        "routing to SMEs, dashboard monitoring, and knowledge base integration. "
        + product_spec
    )
    program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona_program_manager,
        knowledge_program_manager,
    )

    persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
    evaluation_criteria_program_manager = (
        "The answer must contain at least 5 Email Router product features. "
        "Every feature must include these exact labels: "
        "Feature Name:, Description:, Key Functionality:, and User Benefit:. "
        "Respond No if any feature is missing any label, uses numbered lists, bullet points, "
        "or section headings instead of the required labels, or is not Email Router-specific."
    )
    program_manager_evaluation_agent = EvaluationAgent(
        openai_api_key,
        persona_program_manager_eval,
        evaluation_criteria_program_manager,
        program_manager_knowledge_agent,
        max_interactions=10,
    )

    persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
    knowledge_dev_engineer = (
        "Development tasks are defined by identifying what needs to be built to implement each user story. "
        "Write at least 6 detailed Email Router engineering tasks based on the product spec below. "
        "Each task MUST use these exact labels on separate lines:\n"
        "Task ID: <unique id such as ER-001>\n"
        "Task Title: <brief description>\n"
        "Related User Story: <copy the full As a [user], I want [action] so that [benefit] sentence>\n"
        "Description: <detailed technical work>\n"
        "Acceptance Criteria: <requirements for completion>\n"
        "Estimated Effort: <time or complexity>\n"
        "Dependencies: <prerequisite tasks or none>\n"
        "Separate each task with a blank line. "
        "Do not use numbered lists, bullet points, or markdown headings. "
        "Tasks must cover email ingestion, NLP classification, routing rules, "
        "response generation, dashboard monitoring, and integration with existing email infrastructure. "
        + product_spec
    )
    development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
        openai_api_key,
        persona_dev_engineer,
        knowledge_dev_engineer,
    )

    persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
    evaluation_criteria_dev_engineer = (
        "The answer must contain at least 6 Email Router engineering tasks. "
        "Every task must include these exact labels: "
        "Task ID:, Task Title:, Related User Story:, Description:, "
        "Acceptance Criteria:, Estimated Effort:, and Dependencies:. "
        "Related User Story must quote a full user story sentence starting with As a, not a feature name. "
        "Respond No if any task is missing any label, uses numbered lists, bullet points, "
        "or section headings instead of the required labels, or is not Email Router-specific."
    )
    development_engineer_evaluation_agent = EvaluationAgent(
        openai_api_key,
        persona_dev_engineer_eval,
        evaluation_criteria_dev_engineer,
        development_engineer_knowledge_agent,
        max_interactions=10,
    )

    def product_manager_support_function(query: str) -> str:
        knowledge_response = product_manager_knowledge_agent.respond(query)
        evaluation_result = product_manager_evaluation_agent.evaluate(query, knowledge_response)
        return evaluation_result["final_response"]

    def program_manager_support_function(query: str) -> str:
        knowledge_response = program_manager_knowledge_agent.respond(query)
        evaluation_result = program_manager_evaluation_agent.evaluate(query, knowledge_response)
        return evaluation_result["final_response"]

    def development_engineer_support_function(query: str) -> str:
        knowledge_response = development_engineer_knowledge_agent.respond(query)
        evaluation_result = development_engineer_evaluation_agent.evaluate(query, knowledge_response)
        return evaluation_result["final_response"]

    routing_agent = RoutingAgent(openai_api_key, [])
    routing_agent.agents = [
        {
            "name": "Product Manager",
            "description": (
                "Define user stories from a product specification for the Email Router project. "
                "Write user stories using the exact sentence format: As a [user], I want [action] so that [benefit]. "
                "Handles workflow steps that say define user stories from the product spec. "
                "Does not define product features and does not define engineering tasks."
            ),
            "func": product_manager_support_function,
        },
        {
            "name": "Program Manager",
            "description": (
                "Responsible for defining product features by grouping related user stories into capabilities. "
                "Creates Email Router features using Feature Name, Description, Key Functionality, and User Benefit labels. "
                "Handles workflow steps about defining product features or grouping user stories into feature groups. "
                "Does not write user stories and does not create engineering tasks."
            ),
            "func": program_manager_support_function,
        },
        {
            "name": "Development Engineer",
            "description": (
                "Responsible for defining detailed engineering tasks for implementing Email Router user stories. "
                "Creates tasks using Task ID, Task Title, Related User Story, Description, "
                "Acceptance Criteria, Estimated Effort, and Dependencies labels. "
                "Handles workflow steps about defining engineering tasks or development implementation work. "
                "Does not write user stories and does not define product features."
            ),
            "func": development_engineer_support_function,
        },
    ]

    return {
        "action_planning_agent": action_planning_agent,
        "routing_agent": routing_agent,
    }


def run_workflow(workflow_prompt: str = DEFAULT_WORKFLOW_PROMPT) -> dict[str, Any]:
    """Execute the Email Router planning workflow and return the generated output."""
    print("\n*** Workflow execution started ***\n")
    print(f"Task to complete in this workflow, workflow_prompt = {workflow_prompt}")

    openai_api_key = load_openai_api_key()
    product_spec = load_product_spec()
    workflow_agents = build_workflow_agents(openai_api_key, product_spec)

    action_planning_agent = workflow_agents["action_planning_agent"]
    routing_agent = workflow_agents["routing_agent"]

    print("\nDefining workflow steps from the workflow prompt")
    workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
    print(f"Extracted workflow steps: {workflow_steps}")

    completed_steps: list[dict[str, str]] = []
    previous_output = None
    user_stories_output = None

    for step in workflow_steps:
        print(f"\n--- Processing step: {step} ---")
        step_type = get_step_type(step)
        step_query = build_step_query(step, previous_output, user_stories_output)
        result = route_step(step_type, step_query, routing_agent)
        completed_steps.append({"type": step_type, "output": result})
        if step_type == "stories":
            user_stories_output = result
        if step_type == "features":
            previous_output = result
        print(f"Step result:\n{result}")

    print("\n*** Workflow execution completed ***\n")
    print("=" * 60)
    print("FINAL EMAIL ROUTER PROJECT PLAN")
    print("=" * 60)
    section_titles = {
        "stories": "USER STORIES",
        "features": "PRODUCT FEATURES",
        "tasks": "ENGINEERING TASKS",
    }
    for item in completed_steps:
        title = section_titles.get(item["type"], item["type"].upper())
        print(f"\n--- {title} ---")
        print(item["output"])

    return {"steps": completed_steps}


def main() -> None:
    """Console entry point for the workflow."""
    run_workflow()


if __name__ == "__main__":
    main()
