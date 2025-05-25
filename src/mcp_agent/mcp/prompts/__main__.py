from mcp_agent.mcp.prompts.prompt_server import main

# This must be here for the console entry points defined in pyproject.toml
# DO NOT REMOVE!

# For the entry point in pyproject.toml
app = main

if __name__ == "__main__":
    main()
