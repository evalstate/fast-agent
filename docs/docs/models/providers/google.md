## Google

Google is natively supported in `fast-agent` using the Google genai libraries.

**YAML Configuration:**

```yaml
google:
  api_key: "your_google_key"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
```

**Environment Variables:**

- `GOOGLE_API_KEY`: Your Google API key

**Model Name Aliases:**

--8<-- "_generated/model_aliases_google.md"

### OpenAI Mode

You can also access Google via the OpenAI Provider. Use `googleoai` in the YAML file, or `GOOGLEOAI_API_KEY` for API KEY access.
