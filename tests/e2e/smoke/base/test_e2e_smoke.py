# integration_tests/mcp_agent/test_agent_with_image.py
import os
from enum import Enum
from typing import TYPE_CHECKING, List

import pytest
from pydantic import BaseModel, Field

from mcp_agent.core.prompt import Prompt

if TYPE_CHECKING:
    from mcp_agent.llm.memory import Memory
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-mini",
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
        "deepseek",
        #        "generic.qwen2.5:latest",
        #        "generic.llama3.2:latest",
        "openrouter.google/gemini-2.0-flash-001",
        "googleoai.gemini-2.5-flash-preview-05-20",
        "google.gemini-2.0-flash",
        "gemini2",
        "gemini25",  # Works -> Done. Works most of the time, unless Gemini decides to write very long outputs.
        "azure.gpt-4.1",
    ],
)
async def test_basic_textual_prompting(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.send(Prompt.user("write a 50 word story about cats"))
            response_text = response.strip()
            words = response_text.split()
            word_count = len(words)
            assert 40 <= word_count <= 60, f"Expected between 40-60 words, got {word_count}"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1-nano",
    ],
)
async def test_open_ai_history(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="SYSTEM PROMPT",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            await agent.send("MESSAGE ONE")
            await agent.send("MESSAGE TWO")

            provider_history: Memory = agent.agent._llm.history
            multipart_history = agent.agent.message_history

            assert 4 == len(provider_history.get())
            assert 4 == len(multipart_history)

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
        "deepseek",
        "openrouter.google/gemini-2.0-flash-001",
        "gemini2",
        "gemini25",  # Works -> DONE.
        "o3-mini.low",
    ],
)
async def test_multiple_text_blocks_prompting(fast_agent, model_name):
    fast = fast_agent

    # Define the agent
    @fast.agent(
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response: PromptMessageMultipart = await agent.default.generate(
                [Prompt.user("write a 50 word story", "about cats - including the word 'cat'")]
            )
            response_text = response.all_text()
            words = response_text.split()
            word_count = len(words)
            assert 40 <= word_count <= 60, f"Expected between 40-60 words, got {word_count}"
            assert "cat" in response_text

            response: PromptMessageMultipart = await agent.default.generate(
                [
                    Prompt.user("write a 50 word story"),
                    Prompt.user("about cats - including the word 'cat'"),
                ]
            )
            response_text = response.all_text()
            words = response_text.split()
            word_count = len(words)
            assert 40 <= word_count <= 60, f"Expected between 40-60 words, got {word_count}"
            assert "cat" in response_text

    await agent_function()


# Option 2: Using Enum (if you need a proper class)
class WeatherCondition(str, Enum):
    """Possible weather conditions."""

    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    STORMY = "stormy"


# Or as an Enum:
class TemperatureUnit(str, Enum):
    """Temperature unit."""

    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class DailyForecast(BaseModel):
    """Daily weather forecast data."""

    day: str = Field(..., description="Day of the week")
    condition: WeatherCondition = Field(..., description="Weather condition")
    temperature_high: float = Field(..., description="Highest temperature for the day")
    temperature_low: float = Field(..., description="Lowest temperature for the day")
    precipitation_chance: float = Field(..., description="Chance of precipitation (0-100%)")
    notes: str = Field(..., description="Additional forecast notes")


class WeatherForecast(BaseModel):
    """Complete weather forecast with daily data."""

    location: str = Field(..., description="City and country")
    unit: TemperatureUnit = Field(..., description="Temperature unit")
    forecast: List[DailyForecast] = Field(..., description="Daily forecasts")
    summary: str = Field(..., description="Brief summary of the overall forecast")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",  # OpenAI model
        "o3-mini.low",  # reasoning
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gemini2",
        "gemini25",  # Works -> DONE.
        "azure.gpt-4.1",
    ],
)
async def test_structured_weather_forecast_openai_structured_api(fast_agent, model_name):
    """Test that the agent can generate structured weather forecast data."""
    fast = fast_agent

    @fast.agent(
        "weatherforecast",
        instruction="You are a helpful assistant that provides syntehsized weather data for testing purposes.",
        model=model_name,
    )
    async def weather_forecast():
        async with fast.run() as agent:
            # Create a structured prompt that asks for weather forecast
            prompt_text = """
            Generate a 5-day weather forecast for San Francisco, California.
            
            The forecast should include:
            - Daily high and low temperatures in celsius
            - Weather conditions (sunny, cloudy, rainy, snowy, or stormy)
            - Precipitation chance
            - Any special notes about the weather for each day
            
            Provide a brief summary of the overall forecast period at the end.
            """

            # Get structured response
            forecast, result = await agent.weatherforecast.structured(
                [Prompt.user(prompt_text)], WeatherForecast
            )

            # Verify the structured response
            assert forecast is not None, "Structured response should not be None"
            assert isinstance(forecast, WeatherForecast), (
                "Response should be a WeatherForecast object"
            )

            # Verify forecast content
            assert forecast.location.lower().find("san francisco") >= 0, (
                "Location should be San Francisco"
            )
            assert forecast.unit == "celsius", "Temperature unit should be celsius"
            assert len(forecast.forecast) == 5, "Should have 5 days of forecast"
            assert all(isinstance(day, DailyForecast) for day in forecast.forecast), (
                "Each day should be a DailyForecast"
            )

            # Verify data types and ranges
            for day in forecast.forecast:
                assert 0 <= day.precipitation_chance <= 100, (
                    f"Precipitation chance should be 0-100%, got {day.precipitation_chance}"
                )
                assert -50 <= day.temperature_low <= 60, (
                    f"Temperature low should be reasonable, got {day.temperature_low}"
                )
                assert -30 <= day.temperature_high <= 70, (
                    f"Temperature high should be reasonable, got {day.temperature_high}"
                )
                assert day.temperature_high >= day.temperature_low, (
                    "High temp should be >= low temp"
                )

            # Print forecast summary for debugging
            print(f"Weather forecast for {forecast.location}: {forecast.summary}")
            assert '"location":' in result.first_text()

    await weather_forecast()


# @pytest.mark.skip(reason="Local Hardware Required")
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        #        "generic.qwen2.5:latest", qwen often produces short stories, take out for current runs
        "generic.llama3.2:latest",
    ],
)
async def test_generic_model_textual_prompting(fast_agent, model_name):
    """Test that the agent can process an image and respond appropriately."""
    fast = fast_agent

    # Define the agent
    @fast.agent(
        "agent",
        instruction="You are a helpful AI Agent",
        model=model_name,
    )
    async def agent_function():
        async with fast.run() as agent:
            response = await agent.send(Prompt.user("write a 50 word story about cats"))
            response_text = response.strip()
            words = response_text.split()
            word_count = len(words)
            assert 40 <= word_count <= 60, f"Expected between 40-60 words, got {word_count}"

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek",
        "haiku35",
        "gpt-4o",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gemini2",
        "openrouter.google/gemini-2.0-flash-001",
        "gemini25",
        "o3-mini.low",
        "azure.gpt-4.1",
    ],
)
async def test_basic_tool_calling(fast_agent, model_name):
    """Test that the agent can generate structured weather forecast data."""
    fast = fast_agent

    @fast.agent(
        "weatherforecast",
        instruction="You are a helpful assistant that provides synthesized weather data for testing"
        " purposes.",
        model=model_name,
        servers=["test_server"],
    )
    async def weather_forecast():
        async with fast.run() as agent:
            # Delete weather_location.txt if it exists
            if os.path.exists("weather_location.txt"):
                os.remove("weather_location.txt")

            assert not os.path.exists("weather_location.txt")

            response = await agent.send(Prompt.user("what is the weather in london"))
            assert "sunny" in response

            # Check that the file exists after response
            assert os.path.exists("weather_location.txt"), (
                "File should exist after response (created by tool call)"
            )

    await weather_forecast()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek",
        "haiku35",
        "gpt-4o",
        "gpt-4.1-mini",
        "gemini2",
        "gemini25",  # Works -> DONE.
        "openrouter.anthropic/claude-3.7-sonnet",
        "azure.gpt-4.1",
    ],
)
async def test_tool_calls_no_args(fast_agent, model_name):
    fast = fast_agent

    @fast.agent(
        "shirt_colour",
        instruction="You are a helpful assistant that provides information on shirt colours.",
        model=model_name,
        servers=["test_server"],
    )
    async def tools_no_args():
        async with fast.run() as agent:
            response = await agent.send(Prompt.user("get the shirt colour"))
            assert "blue" in response

    await tools_no_args()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek",
        "haiku35",
        #       "gpt-4o",
        #      "gpt-4.1",
        #     "gpt-4.1-nano",
        "gpt-4.1-mini",
        "google.gemini-2.0-flash",
        "gemini25",  # Works -> DONE.
        #       "openrouter.anthropic/claude-3.7-sonnet",
    ],
)
async def test_tool_calls_no_args_typescript(fast_agent, model_name):
    """Temporary test to diagnose typescript server issues"""
    pass
    # fast = fast_agent

    # @fast.agent(
    #     "shirt_colour",
    #     instruction="You are a helpful assistant that provides information on shirt colours.",
    #     model=model_name,
    #     servers=["temp_issue_ts"],
    # )
    # async def tools_no_args_typescript():
    #     async with fast.run() as agent:
    #         response = await agent.send(
    #             Prompt.user("tell me the response from the crashtest1 tool")
    #         )
    #         assert "did it work?" in response

    # await tools_no_args_typescript()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek",
        "haiku35",
        "gpt-4.1",
        "google.gemini-2.0-flash",
        "gemini25",  # Works -> DONE.
    ],
)
async def test_server_has_hyphen(fast_agent, model_name):
    """Test that the agent can generate structured weather forecast data."""
    fast = fast_agent

    @fast.agent(
        "shirt_colour",
        instruction="You are a helpful assistant that provides information on shirt colours.",
        model=model_name,
        servers=["hyphen-name"],
    )
    async def server_with_hyphen():
        async with fast.run() as agent:
            response = await agent.send("check the weather in new york")
            assert "sunny" in response

    await server_with_hyphen()
