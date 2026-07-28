(function () {
  "use strict";

  function tokenize(command) {
    var tokens = [];
    var pattern = /"((?:\\.|[^"\\])*)"|'([^']*)'|([^\s]+)/g;
    var match;
    while ((match = pattern.exec(command)) !== null) {
      tokens.push(
        match[1] !== undefined
          ? match[1].replace(/\\"/g, '"')
          : match[2] !== undefined
            ? match[2]
            : match[3]
      );
    }
    return tokens;
  }

  function shellQuote(value) {
    if (/^[A-Za-z0-9_./:?=@,+-]+$/.test(value)) return value;
    return "'" + value.replace(/'/g, "'\"'\"'") + "'";
  }

  function flagValue(tokens, index) {
    var token = tokens[index];
    var equals = token.indexOf("=");
    if (equals > 0) return { value: token.slice(equals + 1), consumed: 0 };
    return { value: tokens[index + 1] || "", consumed: 1 };
  }

  function fastModel(model, variant) {
    if (!model) return "";
    var converted = model;
    if (model.indexOf("openai/") === 0) {
      converted = "responses." + model.slice("openai/".length);
    } else if (model.indexOf("anthropic/") === 0) {
      converted = "anthropic." + model.slice("anthropic/".length);
    } else if (model.indexOf("/") > 0) {
      converted = model.replace("/", ".");
    } else if (/^gpt-/i.test(model)) {
      converted = "responses." + model;
    }
    if (variant && converted.indexOf("?") < 0) converted += "?reasoning=" + variant;
    return converted;
  }

  function parseClaude(tokens) {
    var result = { source: "Claude Code", prompt: "", model: "", attachments: [] };
    for (var index = 1; index < tokens.length; index += 1) {
      var token = tokens[index];
      if (token === "-p" || token === "--print") {
        if (tokens[index + 1] && tokens[index + 1].indexOf("-") !== 0) {
          result.prompt = tokens[index + 1];
          index += 1;
        }
      } else if (token === "--model" || token.indexOf("--model=") === 0) {
        var model = flagValue(tokens, index);
        result.model = model.value;
        index += model.consumed;
      } else if (token.indexOf("-") !== 0 && !result.prompt) {
        result.prompt = token;
      }
    }
    return result;
  }

  function parseCodex(tokens) {
    var result = {
      source: "Codex",
      prompt: "",
      model: "",
      workspace: "",
      schema: "",
      attachments: [],
    };
    var start = tokens[1] === "exec" ? 2 : 1;
    for (var index = start; index < tokens.length; index += 1) {
      var token = tokens[index];
      if (token === "--model" || token === "-m" || token.indexOf("--model=") === 0) {
        var model = flagValue(tokens, index);
        result.model = fastModel(model.value);
        index += model.consumed;
      } else if (token === "--output-schema" || token.indexOf("--output-schema=") === 0) {
        var schema = flagValue(tokens, index);
        result.schema = schema.value;
        index += schema.consumed;
      } else if (token === "--cd" || token === "-C" || token.indexOf("--cd=") === 0) {
        var workspace = flagValue(tokens, index);
        result.workspace = workspace.value;
        index += workspace.consumed;
      } else if (token.indexOf("-") !== 0) {
        result.prompt = result.prompt ? result.prompt + " " + token : token;
      }
    }
    return result;
  }

  function parseOpenCode(tokens) {
    var result = {
      source: "OpenCode",
      prompt: "",
      model: "",
      variant: "",
      workspace: "",
      agent: "",
      schema: "",
      attachments: [],
    };
    var start = tokens[1] === "run" ? 2 : 1;
    for (var index = start; index < tokens.length; index += 1) {
      var token = tokens[index];
      if (token === "--model" || token === "-m" || token.indexOf("--model=") === 0) {
        var model = flagValue(tokens, index);
        result.model = model.value;
        index += model.consumed;
      } else if (token === "--variant" || token.indexOf("--variant=") === 0) {
        var variant = flagValue(tokens, index);
        result.variant = variant.value;
        index += variant.consumed;
      } else if (token === "--file" || token === "-f" || token.indexOf("--file=") === 0) {
        var attachment = flagValue(tokens, index);
        result.attachments.push(attachment.value);
        index += attachment.consumed;
      } else if (token === "--dir" || token.indexOf("--dir=") === 0) {
        var workspace = flagValue(tokens, index);
        result.workspace = workspace.value;
        index += workspace.consumed;
      } else if (token === "--agent" || token.indexOf("--agent=") === 0) {
        var agent = flagValue(tokens, index);
        result.agent = agent.value;
        index += agent.consumed;
      } else if (token.indexOf("-") !== 0) {
        result.prompt = result.prompt ? result.prompt + " " + token : token;
      }
    }
    result.model = fastModel(result.model, result.variant);
    return result;
  }

  function convert(command) {
    var multiline = /\\\s*\r?\n/.test(command);
    var normalized = command.replace(/\\\s*\r?\n/g, " ");
    var tokens = tokenize(normalized.trim());
    if (tokens.length === 0) throw new Error("Paste a command first.");

    var executable = tokens[0].split("/").pop().toLowerCase();
    var parsed;
    if (executable === "claude") parsed = parseClaude(tokens);
    else if (executable === "codex") parsed = parseCodex(tokens);
    else if (executable === "opencode" || executable === "opencode2") {
      parsed = parseOpenCode(tokens);
    } else {
      throw new Error("Expected a claude, codex, or opencode command.");
    }

    var parts = ["uvx fast-agent-mcp@latest go", "--no-home", "--shell"];
    var model = parsed.model || "";
    if (model) parts.push("--model " + shellQuote(model));
    if (parsed.workspace) parts.push("--workspace " + shellQuote(parsed.workspace));
    if (parsed.agent) parts.push("--agent " + shellQuote(parsed.agent));
    parsed.attachments.forEach(function (attachment) {
      parts.push("--attach " + shellQuote(attachment));
    });
    if (parsed.schema) parts.push("--json-schema " + shellQuote(parsed.schema));
    if (parsed.prompt) parts.push("--message " + shellQuote(parsed.prompt));

    return {
      command: multiline ? parts.join(" \\\n  ") : parts.join(" "),
      source: parsed.source,
    };
  }

  function start() {
    var root = document.querySelector("[data-fa-command-converter]");
    if (!root || root.getAttribute("data-fa-ready") === "true") return;
    root.setAttribute("data-fa-ready", "true");

    var input = root.querySelector("[data-fa-command-input]");
    var output = root.querySelector("[data-fa-command-output]");
    var convertButton = root.querySelector("[data-fa-command-convert]");
    var copyButton = root.querySelector("[data-fa-command-copy]");
    var status = root.querySelector("[data-fa-command-status]");

    convertButton.addEventListener("click", function () {
      try {
        if (!input.value.trim()) input.value = input.placeholder;
        var result = convert(input.value);
        output.textContent = result.command;
        copyButton.disabled = false;
        status.textContent = "Converted from " + result.source + ".";
      } catch (error) {
        output.textContent = "No command generated.";
        copyButton.disabled = true;
        status.textContent = error instanceof Error ? error.message : "Could not convert command.";
      }
    });

    copyButton.addEventListener("click", function () {
      navigator.clipboard.writeText(output.textContent).then(function () {
        status.textContent = "Copied.";
      }, function () {
        status.textContent = "Copy failed. Select the command manually.";
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(start);
})();
