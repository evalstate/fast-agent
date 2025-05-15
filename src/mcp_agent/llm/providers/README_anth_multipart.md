# AnthropicConverter

A utility for converting MCP PromptMessageMultipart messages to Anthropic API format.

## Notes

- The Anthropic API only supports Images and Text in Tool Result blocks (not documents). Other parts are converted to User Messages (similar to OAI)

This converter transforms MCP (Model Context Protocol) user messages to Anthropic's Claude API format. It handles various content types including text, images, and documents while respecting the constraints of Anthropic's API.

## Conversion Rules

| MCP Type                            | Anthropic Type          | Notes                                                           |
| ----------------------------------- | ----------------------- | --------------------------------------------------------------- |
| `TextContent`                       | `text`                  | Supported for user messages                                     |
| `ImageContent`                      | `image`                 | Limited to jpeg, png, gif, webp formats                         |
| `EmbeddedResource` (text)           | `document`              | Converted to text documents with extracted filename as title    |
| `EmbeddedResource` (PDF)            | `document`              | Supported for PDFs only                                         |
| `EmbeddedResource` (image)          | `image`                 | Must be in supported image formats                              |
| `EmbeddedResource` (with image URI) | `image` with URL source | HTTP(S) URIs in image resources are directly used as image URLs |

## Special Cases

- **Unsupported Formats**: Content with unsupported formats (e.g., BMP images) is skipped with a warning
- **Role Restrictions**: Non-text content from assistant role is automatically filtered out
- **URL Resources**: HTTP(S) URLs in resource URIs are used directly with URL source types
- **Missing MIME Types**: When not provided, MIME types are guessed from file extensions
- **Filenames**: Simple filenames and full URIs are both supported for resources
- **Titles**: Document titles are extracted from the filename portion of URIs

## Limitations

- Only supports MIME types allowed by Anthropic's API
- Cannot convert unsupported image formats (only forwards supported formats)
- This converter currently focuses on user message conversions (assistant message conversion handled separately)
