import asyncio

from rich.panel import Panel

from mcp_agent.console import console
from mcp_agent.core.enhanced_prompt import get_enhanced_input, handle_special_commands
from mcp_agent.human_input.types import (
    HumanInputRequest,
    HumanInputResponse,
)
from mcp_agent.progress_display import progress_display


async def console_input_callback(request: HumanInputRequest) -> HumanInputResponse:
    """Request input from a human user via console using prompt_toolkit."""

    # Prepare the prompt text
    prompt_text = request.prompt
    if request.description:
        prompt_text = f"[bold]{request.description}[/bold]\n\n{request.prompt}"

    # Create a panel with the prompt
    panel = Panel(
        prompt_text,
        title="[HUMAN INPUT REQUESTED]",
        title_align="left",
        style="green",
        border_style="bold white",
        padding=(1, 2),
    )

    # Extract agent name from metadata dictionary
    agent_name = (
        request.metadata.get("agent_name", "Unknown Agent") if request.metadata else "Unknown Agent"
    )

    # Use the context manager to pause the progress display while getting input
    with progress_display.paused():
        console.print(panel)

        try:
            if request.timeout_seconds:
                try:
                    # Use get_enhanced_input with empty agent list to disable agent switching
                    response = await asyncio.wait_for(
                        get_enhanced_input(
                            agent_name=agent_name,
                            available_agent_names=[],  # No agents for selection
                            show_stop_hint=False,
                            is_human_input=True,
                            toolbar_color="ansimagenta",
                        ),
                        request.timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    console.print("\n[red]Timeout waiting for input[/red]")
                    raise TimeoutError("No response received within timeout period")
            else:
                response = await get_enhanced_input(
                    agent_name=agent_name,
                    available_agent_names=[],  # No agents for selection
                    show_stop_hint=False,
                    is_human_input=True,
                    toolbar_color="ansimagenta",
                )

            # Handle special commands but ignore dictionary results as they require app context
            command_result = await handle_special_commands(response)
            if isinstance(command_result, dict) and "list_prompts" in command_result:
                from rich import print as rich_print

                rich_print("[yellow]Prompt listing not available in human input context[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Input interrupted[/yellow]")
            response = ""
        except EOFError:
            console.print("\n[yellow]Input terminated[/yellow]")
            response = ""

    return HumanInputResponse(request_id=request.request_id, response=response.strip())
