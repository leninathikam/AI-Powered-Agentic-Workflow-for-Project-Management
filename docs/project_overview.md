# Project Overview

This repository implements a two-stage agentic workflow for the Email Router project.

Phase 1 built the reusable agent toolkit in [src/email_router_workflow/workflow_agents/base_agents.py](../src/email_router_workflow/workflow_agents/base_agents.py). The library includes the direct, persona-based, knowledge-grounded, evaluation, routing, retrieval-augmented, and action-planning agents used by the project.

Phase 2 turns that toolkit into a runnable workflow in [src/email_router_workflow/workflow.py](../src/email_router_workflow/workflow.py). The workflow reads the Email Router product specification from [data/product_specs/email_router.txt](../data/product_specs/email_router.txt), decomposes the request into planning steps, routes each step to the appropriate specialist agent, and produces a structured plan made up of user stories, product features, and engineering tasks.

Reference material from the original starter project has been archived under [archive/legacy_course_materials](../archive/legacy_course_materials), while historical outputs are preserved in [artifacts](../artifacts).
---