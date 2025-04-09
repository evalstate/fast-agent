# integration_tests/mcp_agent/test_agent_with_image.py
import os
from enum import Enum
from typing import List

import pytest
from pydantic import BaseModel, Field

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",  # OpenAI model
        "haiku35",  # Anthropic model
        "deepseek",
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek",  # TODO DeepSeek model - switch to using JSON mode : https://api-docs.deepseek.com/guides/json_mode
        "haiku35",  # TODO Anthropic model - switch to tool calling
    ],
)
async def test_structured_weather_forecast_prompting_style(fast_agent, model_name):
    """Test that the agent can generate structured weather forecast data."""
    fast = fast_agent

    @fast.agent(
        "weatherforecast",
        instruction="You are a helpful assistant that provides syntehsized weather data for testing"
        " purposes.",
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

            Your response must be supplied in JSON format, with the following structure:
            {
                "location": "San Francisco, California",
                "unit": "celsius",
                "forecast": [
                    {
                        "day": "Monday",
                        "condition": "sunny",
                        "temperature_high": 25.0,
                        "temperature_low": 15.0,
                        "precipitation_chance": 10.0,
                        "notes": "A beautiful day ahead."
                    },
                    ...
                ],
                "summary": "Overall, the week looks sunny with a chance of rain on Wednesday."
            }

            DO NOT USE CODE FENCES, BLOCKS, BACKTICKS (`) OR ANY OTHER MARKUP TO FORMAT THE JSON
            """

            # Get structured response
            forecast, _ = await agent.weatherforecast.structured(
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

    await weather_forecast()


@pytest.mark.skip(reason="Generic OpenAI endpoint")
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_name",
    [
        "generic.qwen2.5:latest",
        "generic.llama3.2:latest",
        "openrouter.meta-llama/llama-4-maverick:free",
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
    ["deepseek", "haiku35", "gpt-4o"],
)
async def test_basic_tool_calling(fast_agent, model_name):
    """Test that the agent can generate structured weather forecast data."""
    fast = fast_agent

    @fast.agent(
        "weatherforecast",
        instruction="You are a helpful assistant that provides syntehsized weather data for testing"
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
