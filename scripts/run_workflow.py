"""Command-line launcher for the Email Router workflow."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from email_router_workflow.workflow import main


if __name__ == "__main__":
    main()
