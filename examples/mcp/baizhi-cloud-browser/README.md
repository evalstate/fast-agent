# Baizhi Cloud Browser management MCP

[Baizhi Cloud Browser](https://baizhibrowser.cloud/), also known as CapyBrowser, provides hosted MCP tools for cloud browser runs, saved browser state, human handoff, and downloads. Navigation, clicks, and reading pages need a separate compatible CDP client available to your agent; this configuration only adds management MCP.

## Connect on demand

1. Install fast-agent using its normal setup instructions. Obtain a browser-service key from the provider's [setup guide](https://capybrowser.app.baizhi.cloud/docs/installation.md). Baizhi Cloud Browser and Baizhi Cloud Agent Toolkit are separate services; check which service your key authorizes.
2. Set `BAIZHI_BROWSER_API_KEY` locally to your complete key, without the `Bearer` prefix. Use a secret manager or a local shell input; do not put a real key in this example or commit a secrets file. The YAML adds exactly one `Bearer` prefix and the required space: `Authorization: Bearer YOUR_KEY`.
3. From this directory, inspect the redacted configuration:

   ```bash
   fast-agent config show-mcp fast-agent.yaml
   ```

4. Start fast-agent with the example configuration:

   ```bash
   fast-agent go --config-path fast-agent.yaml
   ```

   Configure your preferred model separately before chatting. Model usage has its own credentials and charges. `load_on_start: false` keeps this server disconnected until you explicitly attach it.
5. In the interactive session, run `/connect baizhi-cloud-browser` and inspect the discovered tools. For a first read-only account check, use `browser_auth_check`; do not create a browser run merely to test authentication. Discovery alone does not prove tool permissions or a complete webpage task.

The config uses fast-agent's default automatic protocol negotiation and its explicit Bearer header. No OAuth flow, local browser-service package, or default browser replacement is needed. If merging into your own YAML, preserve other entries and inspect an existing `mcp.servers.baizhi-cloud-browser` before changing it.

## Control costs and clean up

Browser startup and runtime are billed. Check your account and current prices before calling tools that create runs. Review tool permissions and requested actions before allowing billable or state-changing operations. The provider receives tool inputs; page actions reach the visited sites. Review the provider's terms and privacy information before use.

Disconnect this target with `/mcp disconnect baizhi-cloud-browser` before rotating the environment variable; restart with the replacement key and reconnect to inspect discovery. To remove the configuration, delete only `mcp.servers.baizhi-cloud-browser`, preserving other entries. Disconnecting or removing it does not revoke the provider key.

Disconnecting fast-agent does not end cloud browser runs or stop their billing. Explicitly end the runs you created through the browser tools or [console](https://capybrowser.app.baizhi.cloud/), and check resource release and final charges separately. During human handoff, pause agent actions and resume the same run after control returns.

If authentication fails, check the exact `/mcp` URL, key applicability, and complete header. Do not enter only `Bearer` or use `Authorization=Bearer ...` as the header value. If management tools work but page navigation is missing, configure the separate CDP client.

Contributed by `ctcbyzx`, who maintains the Baizhi Cloud Browser integration materials.
