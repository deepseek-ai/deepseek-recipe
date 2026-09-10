"use strict";

const WEATHER_TOOL = {
  name: "get_weather",
  description: "Get the weather for a location",
  parameters: {
    type: "object",
    properties: { location: { type: "string" }, days: { type: "integer" } },
    required: ["location"],
  },
};

const SAMPLE_REASONING = "The user wants to know the weather, so I need to call get_weather.";

const SAMPLES = {
  chat_completions: pretty({
    tools: [{ type: "function", function: WEATHER_TOOL }],
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What's the weather in Beijing today?" },
      {
        role: "assistant",
        reasoning_content: SAMPLE_REASONING,
        tool_calls: [
          {
            id: "call_1",
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ location: "Beijing", days: 3 }),
            },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "Sunny, 25°C" },
      { role: "assistant", content: "It's sunny in Beijing today, 25°C." },
      { role: "user", content: "Now check and compare the weather in Beijing and Shanghai for the next 3 days." },
    ],
  }),
  responses: pretty({
    instructions: "You are a helpful assistant.",
    tools: [{ type: "function", ...WEATHER_TOOL }],
    input: [
      { type: "message", role: "user", content: "What's the weather in Beijing today?" },
      {
        type: "reasoning",
        content: [{ type: "reasoning_text", text: SAMPLE_REASONING }],
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "get_weather",
        arguments: JSON.stringify({ location: "Beijing", days: 3 }),
      },
      { type: "function_call_output", call_id: "call_1", output: "Sunny, 25°C" },
      { type: "message", role: "assistant", content: "It's sunny in Beijing today, 25°C." },
      { type: "message", role: "user", content: "Now check and compare the weather in Beijing and Shanghai for the next 3 days." },
    ],
  }),
  messages: pretty({
    system: "You are a helpful assistant.",
    tools: [{ name: WEATHER_TOOL.name, description: WEATHER_TOOL.description, input_schema: WEATHER_TOOL.parameters }],
    messages: [
      { role: "user", content: "What's the weather in Beijing today?" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: SAMPLE_REASONING },
          { type: "tool_use", id: "toolu_1", name: "get_weather", input: { location: "Beijing", days: 3 } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "Sunny, 25°C" }],
      },
      { role: "assistant", content: [{ type: "text", text: "It's sunny in Beijing today, 25°C." }] },
      { role: "user", content: "Now check and compare the weather in Beijing and Shanghai for the next 3 days." },
    ],
  }),
};

const DEFAULT_FORMAT = "chat_completions";
const EOT = "<｜end▁of▁sentence｜>";
const OUTPUT_SAMPLE = `<think>The user wants to compare the weather in Beijing and Shanghai for the next three days, so I need forecasts for both cities.
</think>
I'll check the weather in Beijing and Shanghai for the next three days.

<｜DSML｜ calls>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="location" string="true">Beijing</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
<｜DSML｜ invoke name="get_weather">
<｜DSML｜ parameter name="location" string="true">Shanghai</｜DSML｜ parameter>
<｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls>${EOT}`;

const tabs = Array.from(document.querySelectorAll(".tab"));
const modeTabs = Array.from(document.querySelectorAll(".mode-tab"));
const input = document.getElementById("input");
const output = document.getElementById("output");
const meta = document.getElementById("meta");
const errorBox = document.getElementById("error");
const tip = document.getElementById("tip");
const modelOutput = document.getElementById("model-output");
const modelOutputPreview = document.getElementById("model-output-preview");
const decodedOutput = document.getElementById("decoded-output");
const decodeMeta = document.getElementById("decode-meta");
const decodeError = document.getElementById("decode-error");
const responseFormat = document.getElementById("response-format");

const state = { mode: "encode", format: DEFAULT_FORMAT, drafts: { ...SAMPLES }, renderId: 0, decodeId: 0 };

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function pick(value, keys) {
  return Object.fromEntries(keys.filter((key) => value?.[key] != null).map((key) => [key, value[key]]));
}

// Select protocol fields explicitly: tool inputs and argument strings are
// payloads, so metadata-like keys inside them must remain untouched.
function decodedContent(format, response) {
  switch (format) {
    case "chat_completions":
      return {
        choices: response.choices.map((choice) => {
          const message = pick(choice.message, ["content", "reasoning_content"]);
          for (const key of ["content", "reasoning_content"]) {
            if (message[key] === "") delete message[key];
          }
          if (choice.message.tool_calls?.length) {
            message.tool_calls = choice.message.tool_calls.map((tool) => ({
              type: tool.type,
              function: pick(tool.function, ["name", "arguments"]),
            }));
          }
          return { message };
        }),
      };
    case "responses":
      return {
        output: response.output.map((item) => {
          const content = pick(item, ["type", "name", "namespace", "arguments", "input"]);
          if (item.content) content.content = item.content.map((part) => pick(part, ["type", "text"]));
          return content;
        }),
      };
    case "messages":
      return {
        content: response.content.map((block) => pick(block, ["type", "text", "thinking", "name", "input"])),
      };
    default:
      throw new Error(`Unknown API format: ${format}`);
  }
}

function runMode() {
  if (state.mode === "encode") renderPrompt();
  else decodeOutput();
}

function setMode(mode) {
  state.mode = mode;
  for (const tab of modeTabs) {
    const selected = tab.dataset.mode === mode;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  for (const name of ["encode", "decode"]) {
    document.getElementById(`${name}-panel`).hidden = name !== mode;
    document.getElementById(`${name}-actions`).hidden = name !== mode;
  }
  tip.hidden = true;
  runMode();
}

function setFormat(format) {
  state.drafts[state.format] = input.value;
  state.format = format;
  for (const tab of tabs) {
    const selected = tab.dataset.format === format;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected) responseFormat.textContent = tab.textContent;
  }
  input.value = state.drafts[format];
  state.renderId++;
  state.decodeId++;
  clearPrompt();
  clearDecoded();
  hideError();
  runMode();
}

async function renderPrompt() {
  const id = ++state.renderId;
  const format = state.format;
  clearPrompt();
  hideError();
  meta.textContent = "Rendering…";
  output.setAttribute("aria-busy", "true");

  try {
    const body = requestBody();
    const payload = await post("/api/render", { format, body });
    if (id !== state.renderId) return;
    render(payload);
  } catch (err) {
    if (id !== state.renderId) return;
    clearPrompt();
    showError(err.message);
  } finally {
    if (id === state.renderId) output.setAttribute("aria-busy", "false");
  }
}

function requestBody() {
  try {
    return JSON.parse(input.value);
  } catch (err) {
    throw new Error(`Failed to parse request JSON: ${err.message}`);
  }
}

async function post(url, body) {
  let response;
  let payload;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    payload = await response.json();
  } catch (err) {
    throw new Error(`Demo server request failed: ${err.message}`);
  }
  if (!response.ok) throw new Error(payload?.error?.message ?? `HTTP ${response.status}`);
  return payload;
}

async function decodeOutput() {
  const id = ++state.decodeId;
  const format = state.format;
  clearDecoded();
  decodeMeta.textContent = "Decoding…";
  decodedOutput.setAttribute("aria-busy", "true");
  try {
    const body = requestBody();
    const payload = await post("/api/decode", {
      format,
      body,
      output: modelOutput.value,
    });
    if (id !== state.decodeId) return;
    renderModelOutput(payload.segments);
    decodedOutput.textContent = pretty(decodedContent(format, payload.response));
    decodeMeta.textContent = "";
  } catch (err) {
    if (id !== state.decodeId) return;
    clearDecoded();
    decodeError.textContent = err.message;
    decodeError.hidden = false;
  } finally {
    if (id === state.decodeId) decodedOutput.setAttribute("aria-busy", "false");
  }
}

function clearPrompt() {
  output.replaceChildren();
  output.setAttribute("aria-busy", "false");
  meta.textContent = "";
  tip.hidden = true;
}

function clearDecoded() {
  decodedOutput.textContent = "";
  decodedOutput.setAttribute("aria-busy", "false");
  decodeMeta.textContent = "";
  decodeError.hidden = true;
}

function invalidateDecoded() {
  state.decodeId++;
  clearDecoded();
  decodeMeta.textContent = "Input changed, ready to decode";
}

function render(payload) {
  const fragment = document.createDocumentFragment();
  for (const segment of payload.segments) {
    fragment.append(renderSegment(segment));
  }
  output.replaceChildren(fragment);
  meta.textContent = `${payload.prompt.length} chars`;
}

function syncModelOutputScroll() {
  modelOutputPreview.scrollTop = modelOutput.scrollTop;
  modelOutputPreview.scrollLeft = modelOutput.scrollLeft;
}

function renderModelOutput(segments) {
  if (segments) {
    const fragment = document.createDocumentFragment();
    for (const segment of segments) fragment.append(renderSegment(segment));
    modelOutputPreview.replaceChildren(fragment);
  } else {
    modelOutputPreview.textContent = modelOutput.value;
  }
  if (modelOutput.value.endsWith("\n")) {
    modelOutputPreview.append(document.createTextNode("\u200b"));
  }
  syncModelOutputScroll();
}

function renderSegment(segment) {
  const span = document.createElement("span");
  if (segment.kind === "token") {
    span.className = "sp";
    span.textContent = segment.token;
    span.dataset.name = segment.name;
    span.dataset.tip = segment.description;
  } else {
    span.className = "tx";
    span.textContent = segment.text;
  }
  return span;
}

function positionTip(event) {
  const margin = 12;
  tip.style.left = "0px";
  tip.style.top = "0px";
  const rect = tip.getBoundingClientRect();
  let x = event.clientX + margin;
  let y = event.clientY + margin;
  if (x + rect.width > window.innerWidth - margin) {
    x = Math.max(margin, window.innerWidth - margin - rect.width);
  }
  if (y + rect.height > window.innerHeight - margin) {
    y = Math.max(margin, event.clientY - margin - rect.height);
  }
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

function bindTabs(buttons, select) {
  for (const tab of buttons) {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      const index = buttons.indexOf(tab);
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % buttons.length;
      if (event.key === "ArrowLeft") next = (index + buttons.length - 1) % buttons.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = buttons.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      buttons[next].focus();
      select(buttons[next]);
    });
  }
}

bindTabs(tabs, (tab) => setFormat(tab.dataset.format));
bindTabs(modeTabs, (tab) => setMode(tab.dataset.mode));

output.addEventListener("mouseover", (event) => {
  const token = event.target.closest(".sp");
  if (!token) return;
  tip.textContent = token.dataset.tip;
  tip.hidden = false;
  positionTip(event);
});

output.addEventListener("mousemove", (event) => {
  if (!tip.hidden) positionTip(event);
});

output.addEventListener("mouseout", (event) => {
  if (event.target.closest(".sp")) tip.hidden = true;
});

document.getElementById("render-prompt").addEventListener("click", renderPrompt);
document.getElementById("decode-output").addEventListener("click", decodeOutput);

input.addEventListener("input", () => {
  state.drafts[state.format] = input.value;
  state.renderId++;
  clearPrompt();
  hideError();
  meta.textContent = "Request changed, ready to render";
  invalidateDecoded();
});

modelOutput.addEventListener("input", () => {
  renderModelOutput();
  invalidateDecoded();
});
modelOutput.addEventListener("scroll", syncModelOutputScroll);
new ResizeObserver(syncModelOutputScroll).observe(modelOutput);

modelOutput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    decodeOutput();
  }
});

input.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    renderPrompt();
  }
});

input.value = state.drafts[DEFAULT_FORMAT];
modelOutput.value = OUTPUT_SAMPLE;
renderModelOutput();
setFormat(DEFAULT_FORMAT);
