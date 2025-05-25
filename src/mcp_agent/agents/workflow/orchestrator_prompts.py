"""
Prompt templates used by the Orchestrator workflow.
"""

# Templates for formatting results
TASK_RESULT_TEMPLATE = """Task: {task_description}
Result: {task_result}"""

STEP_RESULT_TEMPLATE = """Step: {step_description}
Step Subtasks:
{tasks_str}"""

PLAN_RESULT_TEMPLATE = """Plan Objective: {plan_objective}

Progress So Far (steps completed):
{steps_str}

Result: {plan_result}"""

FULL_PLAN_PROMPT_TEMPLATE = """You are tasked with orchestrating a plan to complete an objective.
You can analyze results from the previous steps already executed to decide if the objective is complete.

<fastagent:data>
<fastagent:objective>
{objective}
</fastagent:objective>

<fastagent:available-agents>
{agents}
</fastagent:available-agents>

<fastagent:progress>
{plan_result}
</fastagent:progress>

<fastagent:status>
{plan_status}
{iterations_info}
</fastagent:status>
</fastagent:data>

Your plan must be structured in sequential steps, with each step containing independent parallel subtasks.
If the previous results achieve the objective, return is_complete=True.
Otherwise, generate remaining steps needed.

<fastagent:instruction>
You are operating in "full plan" mode, where you generate a complete plan with ALL remaining steps needed.
After receiving your plan, the system will execute ALL steps in your plan before asking for your input again.
If the plan needs multiple iterations, you'll be called again with updated results.

Generate a plan with all remaining steps needed.
Steps are sequential, but each Step can have parallel subtasks.
For each Step, specify a description of the step and independent subtasks that can run in parallel.
For each subtask specify:
    1. Clear description of the task that an LLM can execute  
    2. Name of 1 Agent from the available agents list above
    
CRITICAL: You MUST ONLY use agent names that are EXACTLY as they appear in <fastagent:available-agents> above.
Do NOT invent new agents. Do NOT modify agent names. The plan will FAIL if you use an agent that doesn't exist.

Return your response in the following JSON structure:
    {{
        "steps": [
            {{
                "description": "Description of step 1",
                "tasks": [
                    {{
                        "description": "Description of task 1",
                        "agent": "agent_name"  // agent MUST be exactly one of the agent names listed above
                    }},
                    {{
                        "description": "Description of task 2", 
                        "agent": "agent_name2"  // agent MUST be exactly one of the agent names listed above
                    }}
                ]
            }}
        ],
        "is_complete": false
    }}

Set "is_complete" to true when ANY of these conditions are met:
1. The objective has been achieved in full or substantively
2. The remaining work is minor or trivial compared to what's been accomplished
3. Additional steps provide minimal value toward the core objective
4. The plan has gathered sufficient information to answer the original request

Be decisive - avoid excessive planning steps that add little value. It's better to complete a plan early than to continue with marginal improvements. Focus on the core intent of the objective, not perfection.

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences.
</fastagent:instruction>
"""

ITERATIVE_PLAN_PROMPT_TEMPLATE = """You are tasked with determining only the next step in a plan
needed to complete an objective. You must analyze the current state and progress from previous steps 
to decide what to do next.

<fastagent:data>
<fastagent:objective>
{objective}
</fastagent:objective>

<fastagent:available-agents>
{agents}
</fastagent:available-agents>

<fastagent:progress>
{plan_result}
</fastagent:progress>

<fastagent:status>
{plan_status}
{iterations_info}
</fastagent:status>
</fastagent:data>

A Step must be sequential in the plan, but can have independent parallel subtasks. Only return a single Step.
If the previous results achieve the objective, return is_complete=True.
Otherwise, generate the next Step.

<fastagent:instruction>
You are operating in "iterative plan" mode, where you generate ONLY ONE STEP at a time.
After each step is executed, you'll be called again to determine the next step based on updated results.

Generate the next step, by specifying a description of the step and independent subtasks that can run in parallel:
For each subtask specify:
    1. Clear description of the task that an LLM can execute  
    2. Name of 1 Agent from the available agents list above

CRITICAL: You MUST ONLY use agent names that are EXACTLY as they appear in <fastagent:available-agents> above.
Do NOT invent new agents. Do NOT modify agent names. The plan will FAIL if you use an agent that doesn't exist.

Return your response in the following JSON structure:
    {{
        "description": "Description of step 1",
        "tasks": [
            {{
                "description": "Description of task 1",
                "agent": "agent_name"  // agent MUST be exactly one of the agent names listed above
            }}
        ],
        "is_complete": false
    }}

Set "is_complete" to true when ANY of these conditions are met:
1. The objective has been achieved in full or substantively
2. The remaining work is minor or trivial compared to what's been accomplished
3. Additional steps provide minimal value toward the core objective
4. The plan has gathered sufficient information to answer the original request

Be decisive - avoid excessive planning steps that add little value. It's better to complete a plan early than to continue with marginal improvements. Focus on the core intent of the objective, not perfection.

You must respond with valid JSON only, with no triple backticks. No markdown formatting.
No extra text. Do not wrap in ```json code fences.
</fastagent:instruction>
"""

TASK_PROMPT_TEMPLATE = """You are part of a larger workflow to achieve an objective.

<fastagent:data>
<fastagent:objective>
{objective}
</fastagent:objective>

<fastagent:task>
{task}
</fastagent:task>

<fastagent:context>
{context}
</fastagent:context>
</fastagent:data>

<fastagent:instruction>
Your job is to accomplish only the task specified above.
Use the context from previous steps to inform your approach.
The context contains structured XML with the results from previous steps - pay close attention to:
- The objective in <fastagent:objective>
- Previous step results in <fastagent:steps>
- Task results and their attribution in <fastagent:task-result>

Provide a direct, focused response that addresses the task.
</fastagent:instruction>
"""

SYNTHESIZE_STEP_PROMPT_TEMPLATE = """You need to synthesize the results of parallel tasks into a cohesive result.

<fastagent:data>
<fastagent:step-results>
{step_result}
</fastagent:step-results>
</fastagent:data>

<fastagent:instruction>
Analyze the results from all tasks in this step.
Each task was executed by a specific agent (finder, writer, etc.)
Consider the expertise of each agent when weighing their results.
Combine the information into a coherent, unified response.
Focus on key insights and important outcomes.
Resolve any conflicting information if present.
</fastagent:instruction>
"""

SYNTHESIZE_PLAN_PROMPT_TEMPLATE = """You need to synthesize the results of all completed plan steps into a final response.

<fastagent:data>
<fastagent:plan-results>
{plan_result}
</fastagent:plan-results>
</fastagent:data>

<fastagent:instruction>
Create a comprehensive final response that addresses the original objective.
Integrate all the information gathered across all plan steps.
Provide a clear, complete answer that achieves the objective.
Focus on delivering value through your synthesis, not just summarizing.

If the plan was marked as incomplete but the maximum number of iterations was reached,
make sure to state clearly what was accomplished and what remains to be done.
</fastagent:instruction>
"""

# New template for incomplete plans due to iteration limits
SYNTHESIZE_INCOMPLETE_PLAN_TEMPLATE = """You need to synthesize the results of all completed plan steps into a final response.

<fastagent:data>
<fastagent:plan-results>
{plan_result}
</fastagent:plan-results>
</fastagent:data>

<fastagent:status>
The maximum number of iterations ({max_iterations}) was reached before the objective could be completed.
</fastagent:status>

<fastagent:instruction>
Create a comprehensive response that summarizes what was accomplished so far.
The objective was NOT fully completed due to reaching the maximum number of iterations.

In your response:
1. Clearly state that the objective was not fully completed
2. Summarize what WAS accomplished across all the executed steps
3. Identify what remains to be done to complete the objective
4. Organize the information to provide maximum value despite being incomplete

Focus on being transparent about the incomplete status while providing as much value as possible.
</fastagent:instruction>
"""
