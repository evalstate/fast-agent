"""
Prompt templates used by the Orchestrator workflow.
"""

# Templates for formatting results
TASK_RESULT_TEMPLATE = """Task: {task_description}
Result: {task_result}"""

STEP_RESULT_TEMPLATE = """Step: {step_description}
Step Subtasks:
{tasks_str}"""
