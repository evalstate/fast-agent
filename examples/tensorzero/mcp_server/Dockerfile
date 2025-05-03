# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the dependency files from the project root
# Build context is project root, paths are relative to that
COPY pyproject.toml /app/
COPY uv.lock /app/
COPY LICENSE /app/
COPY README.md /app/

# Install project dependencies using uv
# This installs the 'fast-agent-mcp' package and its dependencies needed by the server
RUN uv pip install --system .

# Copy the MCP server script into the container
# Path is relative to the build context (project root)
COPY examples/tensorzero/mcp_server/mcp_server.py /app/

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run uvicorn when the container launches
# Assumes the Starlette app instance in mcp_server.py is named 'app'
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"] 