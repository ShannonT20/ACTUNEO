"""
Model backends

Every backend offers the same two methods:

``generate(prompt, system=None, max_tokens=None, temperature=0.2)``
    Text in, text out.
``call_tools(prompt, tools, system=None, allow_none=True)``
    Pick one of the given tools and fill in its arguments. Returns a
    :class:`ToolCall`.

:class:`LocalBackend` runs a downloaded GGUF model with ``llama-cpp-python``.
The others call a provider with the user's own key: :class:`OpenAIBackend`,
:class:`AnthropicBackend`, :class:`GeminiBackend` and :class:`OllamaBackend`.

Tools are described the way most providers do it::

    {"name": "life_expectancy",
     "description": "Expectation of life at an age",
     "parameters": {"type": "object",
                    "properties": {"age": {"type": "integer"}},
                    "required": ["age"]}}
"""

import json
import logging
import os
import re
from collections import OrderedDict
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("actuneo.ai")

PROVIDERS = ("gemini", "openai", "anthropic", "ollama")

#: Environment variables read for each provider's key, in order
KEY_VARIABLES = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": (),
}

#: Model used when none is named. Providers rename and retire models: check
#: the provider's current list and pass ``model=`` if a default stops working.
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen2.5:7b",
}

NO_TOOL = "none"


class BackendError(RuntimeError):
    """A backend could not be loaded or a request to it failed."""


@dataclass
class ToolCall:
    """
    The model's choice of tool.

    Attributes:
        name: Name of the tool chosen, or None if the model chose none
        arguments: Arguments for the tool
        text: Any text the model gave, such as its answer when no tool fits
    """
    name: Optional[str]
    arguments: Dict[str, Any] = field(default_factory=dict)
    text: str = ""


def normalise_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tool descriptions in one shape: ``name``, ``description``, ``parameters``.

    Accepts that shape, the OpenAI wrapper ``{"type": "function",
    "function": {...}}`` and Anthropic's ``input_schema`` spelling.
    """
    result = []
    for tool in tools:
        spec = tool.get("function", tool)
        name = spec.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("every tool needs a name")
        if name == NO_TOOL:
            raise ValueError(f"'{NO_TOOL}' is reserved and cannot be a tool name")
        parameters = spec.get("parameters") or spec.get("input_schema") \
            or {"type": "object", "properties": {}}
        result.append({"name": name, "description": spec.get("description", ""),
                       "parameters": parameters})
    if not result:
        raise ValueError("at least one tool is needed")
    names = [tool["name"] for tool in result]
    if len(set(names)) != len(names):
        raise ValueError("tool names must be unique")
    return result


def tool_choice_schema(tools: Sequence[Dict[str, Any]], allow_none: bool = True) -> Dict[str, Any]:
    """
    JSON schema of a valid tool choice: ``{"tool": <name>, "arguments": {...}}``
    for one of the tools, or ``{"tool": "none", "answer": "..."}``.

    Used where the whole choice is made in one constrained reply (Ollama).
    A reply held to this schema cannot name a tool that does not exist or
    give arguments of the wrong shape.
    """
    options = []
    for tool in normalise_tools(tools):
        options.append({
            "type": "object",
            "properties": {"tool": {"type": "string", "enum": [tool["name"]]},
                           "arguments": tool["parameters"]},
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        })
    if allow_none:
        options.append({
            "type": "object",
            "properties": {"tool": {"type": "string", "enum": [NO_TOOL]},
                           "answer": {"type": "string"}},
            "required": ["tool", "answer"],
            "additionalProperties": False,
        })
    return {"anyOf": options}


def _tool_prompt(tools: Sequence[Dict[str, Any]], allow_none: bool) -> str:
    lines = ["You decide whether one of these functions answers the user's request.",
             "Functions:"]
    for tool in tools:
        lines.append(json.dumps(tool))
    lines.append("Rules:")
    if allow_none:
        lines.append(f'- If the request is a greeting, small talk, or something none of the '
                     f'functions does, reply {{"tool": "{NO_TOOL}", "answer": "<your reply>"}}.')
    lines.append('- Otherwise reply {"tool": "<function name>", "arguments": {...}}.')
    lines.append("- Use only values the user gave. Never invent a value.")
    lines.append("Reply with JSON only.")
    return "\n".join(lines)


_WORD = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    "a an and at by for from given how in is it of on or the to what with using use my me i "
    "can you please value values calculate compute find get".split())
_THINK = re.compile(r"<think>.*?</think>\s*", flags=re.S)


def _words(text: str) -> List[str]:
    words = []
    for word in _WORD.findall(text.lower().replace("_", " ")):
        if word in _STOP_WORDS:
            continue
        words.append(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def shortlist_tools(prompt: str, tools: Sequence[Dict[str, Any]],
                    limit: int = 5) -> List[Dict[str, Any]]:
    """
    The tools most relevant to a request, so that only those are put in
    front of the model. Reading a long list of tools is the slowest part of
    a tool call on a processor.

    Relevance is a count of the words the request shares with each tool's
    name and description, the name counting double. It is a plain word
    match, not a search by meaning: a request phrased in words the tools do
    not use can miss the right one, so keep ``limit`` generous or describe
    tools in the words users use.

    Args:
        prompt: The user's request
        tools: All the tools
        limit: How many to keep

    Returns:
        Up to ``limit`` tools, most relevant first. If the request shares no
        word with any tool, the first ``limit`` tools in their given order.
    """
    tools = normalise_tools(tools)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(tools) <= limit:
        return tools
    asked = set(_words(prompt))
    scored = []
    for position, tool in enumerate(tools):
        score = (2 * len(asked & set(_words(tool["name"])))
                 + len(asked & set(_words(tool["description"]))))
        scored.append((-score, position, tool))
    scored.sort(key=lambda item: item[:2])
    return [tool for _, _, tool in scored[:limit]]


def selection_schema(tools: Sequence[Dict[str, Any]], allow_none: bool = True) -> Dict[str, Any]:
    """JSON schema of the first step of a local tool call: which function, or none."""
    names = [tool["name"] for tool in normalise_tools(tools)]
    if allow_none:
        names.append(NO_TOOL)
    return {"type": "object",
            "properties": {"function": {"type": "string", "enum": names}},
            "required": ["function"], "additionalProperties": False}


def _selection_prompt(tools: Sequence[Dict[str, Any]], allow_none: bool) -> str:
    lines = ["Pick the function that does what the user asks. Functions:"]
    lines.extend(f"- {tool['name']}: {tool['description']}" for tool in tools)
    example = tools[0]["name"]
    if allow_none:
        lines.append(f"- {NO_TOOL}: the request is a greeting, small talk, general knowledge, "
                     "or anything these functions do not do")
        example = NO_TOOL
    lines.append(f'Reply with JSON only, for example {{"function": "{example}"}}.')
    return "\n".join(lines)


def _arguments_prompt(tool: Dict[str, Any]) -> str:
    return (f"Fill in the arguments of the function {tool['name']} ({tool['description']}) "
            f"from the user's request.\nArguments: {json.dumps(tool['parameters'])}\n"
            "Use only values the user gave. Leave out optional arguments the user did not "
            "mention. Reply with JSON only.")


def parse_tool_choice(text: str, tools: Sequence[Dict[str, Any]]) -> ToolCall:
    """Turn the JSON a model returned into a :class:`ToolCall`, checking the name."""
    try:
        data = json.loads(text)
    except ValueError as error:
        raise BackendError(f"the model did not return valid JSON: {text[:200]!r}") from error
    if not isinstance(data, dict):
        raise BackendError(f"the model returned JSON that is not an object: {text[:200]!r}")
    name = data.get("tool")
    if name in (None, NO_TOOL):
        return ToolCall(None, {}, str(data.get("answer", "")))
    if name not in [tool["name"] for tool in tools]:
        raise BackendError(f"the model chose an unknown tool '{name}'")
    arguments = data.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise BackendError("the model returned arguments that are not an object")
    return ToolCall(name, arguments)


class Backend(ABC):
    """Interface shared by every backend."""

    #: Short name of the backend, such as "local" or "openai"
    name: str = ""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        """Answer a prompt with text."""

    @abstractmethod
    def call_tools(self, prompt: str, tools: Sequence[Dict[str, Any]],
                   system: Optional[str] = None, allow_none: bool = True) -> ToolCall:
        """Choose one of the tools for a prompt and fill in its arguments."""


# ----------------------------------------------------------------------
# Local model
# ----------------------------------------------------------------------
class LocalBackend(Backend):
    """
    A GGUF model run on this machine with ``llama-cpp-python``.

    The model is loaded on first use. Nothing leaves the machine.
    """

    name = "local"

    def __init__(self, model_path: str, n_gpu_layers: int = 0, n_ctx: int = 4096,
                 n_threads: Optional[int] = None, verbose: bool = False,
                 no_think: bool = False, max_tools: int = 5, cache_size: int = 128):
        """
        Args:
            model_path: Path of the GGUF file
            n_gpu_layers: Layers to run on the GPU: 0 for none, -1 for all
            n_ctx: Context window in tokens. A small window keeps memory
                use and start-up time down; 2,000 to 4,000 is plenty for
                tool calls.
            n_threads: Processor threads. Use the number of physical cores,
                not hyperthreads (chosen by llama.cpp if omitted).
            verbose: Print llama.cpp's own log
            no_think: Add Qwen3's ``/no_think`` switch to every prompt, so
                the model answers without first writing out its reasoning.
                Reasoning is what makes a processor painfully slow.
            max_tools: Most tools put in front of the model in
                ``call_tools``; the rest are dropped by
                :func:`shortlist_tools`
            cache_size: Number of recent answers remembered, so a repeated
                question is answered at once (0 to turn off)
        """
        self.model_path = str(model_path)
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_ctx = int(n_ctx)
        self.n_threads = n_threads
        self.verbose = verbose
        self.no_think = bool(no_think)
        self.max_tools = int(max_tools)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._llm = None

    def clear_cache(self) -> None:
        """Forget the remembered answers."""
        self._cache.clear()

    def _remembered(self, key: str, compute):
        """The remembered answer for a key, or a fresh one that is then remembered."""
        if self.cache_size <= 0:
            return compute()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = compute()
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _model(self):
        if self._llm is None:
            if not os.path.exists(self.model_path):
                raise BackendError(f"model file not found: {self.model_path}. "
                                   "Run actuneo.ai.setup() again.")
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise BackendError("llama-cpp-python is not installed. "
                                   "Install it with: pip install actuneo[ai]") from error
            logger.info("Loading %s (GPU layers %s)", self.model_path, self.n_gpu_layers)
            options = dict(model_path=self.model_path, n_gpu_layers=self.n_gpu_layers,
                           n_ctx=self.n_ctx, verbose=self.verbose)
            if self.n_threads:
                options["n_threads"] = int(self.n_threads)
            self._llm = Llama(**options)
        return self._llm

    @staticmethod
    def _messages(prompt: str, system: Optional[str]) -> List[Dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _chat(self, prompt: str, system: Optional[str], **options) -> str:
        """One exchange with the model, with thinking switched off if asked."""
        if self.no_think:
            prompt = f"{prompt} /no_think"
        result = self._model().create_chat_completion(
            messages=self._messages(prompt, system), **options)
        text = result["choices"][0]["message"].get("content") or ""
        return _THINK.sub("", text).strip()

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        key = json.dumps(["generate", prompt, system, max_tokens, temperature])
        return self._remembered(key, lambda: self._chat(
            prompt, system, max_tokens=max_tokens or 1024, temperature=temperature))

    def _constrained(self, prompt: str, instructions: str, schema: Dict[str, Any],
                     system: Optional[str], max_tokens: int) -> Dict[str, Any]:
        """Ask for JSON that must satisfy a schema."""
        if system:
            instructions = system + "\n\n" + instructions
        text = self._chat(prompt, instructions, max_tokens=max_tokens, temperature=0.0,
                          response_format={"type": "json_object", "schema": schema})
        try:
            data = json.loads(text)
        except ValueError as error:
            raise BackendError(f"the model did not return valid JSON: {text[:200]!r}") from error
        if not isinstance(data, dict):
            raise BackendError(f"the model returned JSON that is not an object: {text[:200]!r}")
        return data

    def call_tools(self, prompt: str, tools: Sequence[Dict[str, Any]],
                   system: Optional[str] = None, allow_none: bool = True) -> ToolCall:
        """
        Choose a tool in two constrained steps, which small models follow far
        more reliably than one: first the function (or none) from a fixed
        list, then that function's arguments, held to its schema. When no
        function fits, the request is answered as ordinary text.

        Only the ``max_tools`` most relevant tools are shown to the model
        (see :func:`shortlist_tools`), and the model writes only a function
        name and its arguments: a few dozen tokens.
        """
        tools = normalise_tools(tools)
        key = json.dumps(["call_tools", prompt, system, allow_none, tools], sort_keys=True)
        return self._remembered(key, lambda: self._choose(prompt, tools, system, allow_none))

    def _choose(self, prompt: str, tools: List[Dict[str, Any]], system: Optional[str],
                allow_none: bool) -> ToolCall:
        tools = shortlist_tools(prompt, tools, self.max_tools)
        chosen = self._constrained(prompt, _selection_prompt(tools, allow_none),
                                   selection_schema(tools, allow_none), system, 40)
        name = chosen.get("function")
        if name == NO_TOOL and allow_none:
            return ToolCall(None, {}, self.generate(prompt, system))
        matches = [tool for tool in tools if tool["name"] == name]
        if not matches:
            raise BackendError(f"the model chose an unknown tool '{name}'")
        arguments = self._constrained(prompt, _arguments_prompt(matches[0]),
                                      matches[0]["parameters"], system, 512)
        return ToolCall(name, arguments)

    def __repr__(self) -> str:
        return f"LocalBackend(model_path='{self.model_path}', n_gpu_layers={self.n_gpu_layers})"


# ----------------------------------------------------------------------
# Providers reached over HTTP
# ----------------------------------------------------------------------
def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any],
               timeout: float = 120.0) -> Dict[str, Any]:
    """POST JSON and return the JSON reply. Keys in headers are never logged."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    context = None
    if url.startswith("https"):
        import ssl
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        if error.code in (401, 403):
            raise BackendError(f"the provider rejected the API key (HTTP {error.code}).") from error
        if error.code == 429:
            raise BackendError("the provider's rate limit or quota was reached (HTTP 429). "
                               "Wait and try again.") from error
        raise BackendError(f"the provider returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise BackendError(f"could not reach {url.split('/')[2]}: {error}. "
                           "Check the internet connection.") from error


class OpenAIBackend(Backend):
    """OpenAI's chat completions API, or any server that speaks the same protocol."""

    name = "openai"

    def __init__(self, api_key: str, model: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1"):
        self._key = api_key
        self.model = model or DEFAULT_MODELS["openai"]
        self.base_url = base_url.rstrip("/")

    def _chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        reply = _post_json(f"{self.base_url}/chat/completions",
                           {"Authorization": f"Bearer {self._key}"},
                           {"model": self.model, **body})
        return reply["choices"][0]["message"]

    def generate(self, prompt, system=None, max_tokens=None, temperature=0.2) -> str:
        body: Dict[str, Any] = {"messages": LocalBackend._messages(prompt, system)}
        if max_tokens:
            body["max_completion_tokens"] = int(max_tokens)
        return (self._chat(body).get("content") or "").strip()

    def call_tools(self, prompt, tools, system=None, allow_none=True) -> ToolCall:
        tools = normalise_tools(tools)
        message = self._chat({
            "messages": LocalBackend._messages(prompt, system),
            "tools": [{"type": "function", "function": tool} for tool in tools],
            "tool_choice": "auto" if allow_none else "required",
        })
        calls = message.get("tool_calls") or []
        if not calls:
            return ToolCall(None, {}, (message.get("content") or "").strip())
        function = calls[0]["function"]
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return ToolCall(function["name"], arguments)

    def __repr__(self) -> str:
        return f"OpenAIBackend(model='{self.model}')"


class GeminiBackend(Backend):
    """Google's Gemini API."""

    name = "gemini"

    def __init__(self, api_key: str, model: Optional[str] = None,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        self._key = api_key
        self.model = model or DEFAULT_MODELS["gemini"]
        self.base_url = base_url.rstrip("/")

    def _request(self, body: Dict[str, Any], system: Optional[str]) -> List[Dict[str, Any]]:
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        reply = _post_json(f"{self.base_url}/models/{self.model}:generateContent",
                           {"x-goog-api-key": self._key}, body)
        candidates = reply.get("candidates") or []
        if not candidates:
            raise BackendError(f"Gemini returned no answer: {json.dumps(reply)[:300]}")
        return candidates[0].get("content", {}).get("parts") or []

    def generate(self, prompt, system=None, max_tokens=None, temperature=0.2) -> str:
        config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            config["maxOutputTokens"] = int(max_tokens)
        parts = self._request({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                               "generationConfig": config}, system)
        return "".join(part.get("text", "") for part in parts).strip()

    def call_tools(self, prompt, tools, system=None, allow_none=True) -> ToolCall:
        tools = normalise_tools(tools)
        parts = self._request({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"functionDeclarations": tools}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO" if allow_none else "ANY"}},
        }, system)
        for part in parts:
            if "functionCall" in part:
                call = part["functionCall"]
                return ToolCall(call["name"], dict(call.get("args") or {}))
        return ToolCall(None, {}, "".join(part.get("text", "") for part in parts).strip())

    def __repr__(self) -> str:
        return f"GeminiBackend(model='{self.model}')"


class OllamaBackend(Backend):
    """A model served by Ollama on this machine or the local network. No key is needed."""

    name = "ollama"

    def __init__(self, model: Optional[str] = None, base_url: str = "http://localhost:11434"):
        self.model = model or DEFAULT_MODELS["ollama"]
        self.base_url = base_url.rstrip("/")

    def _chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        reply = _post_json(f"{self.base_url}/api/chat", {},
                           {"model": self.model, "stream": False, **body}, timeout=600)
        return reply.get("message", {})

    def generate(self, prompt, system=None, max_tokens=None, temperature=0.2) -> str:
        options: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        message = self._chat({"messages": LocalBackend._messages(prompt, system),
                              "options": options})
        return (message.get("content") or "").strip()

    def call_tools(self, prompt, tools, system=None, allow_none=True) -> ToolCall:
        tools = normalise_tools(tools)
        instructions = _tool_prompt(tools, allow_none)
        if system:
            instructions = system + "\n\n" + instructions
        message = self._chat({
            "messages": LocalBackend._messages(prompt, instructions),
            "format": tool_choice_schema(tools, allow_none),
            "options": {"temperature": 0},
        })
        return parse_tool_choice(message.get("content") or "", tools)

    def __repr__(self) -> str:
        return f"OllamaBackend(model='{self.model}')"


# ----------------------------------------------------------------------
# Anthropic, through its official SDK
# ----------------------------------------------------------------------
def _strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Close object schemas, as strict tool use requires."""
    if not isinstance(schema, dict):
        return schema
    closed = {key: _strict_schema(value) if isinstance(value, dict) else value
              for key, value in schema.items()}
    if "properties" in closed:
        closed["properties"] = {name: _strict_schema(sub)
                                for name, sub in closed["properties"].items()}
    if isinstance(closed.get("items"), dict):
        closed["items"] = _strict_schema(closed["items"])
    if closed.get("type") == "object":
        closed.setdefault("additionalProperties", False)
        closed.setdefault("required", list(closed.get("properties", {})))
    return closed


class AnthropicBackend(Backend):
    """
    Claude, through the ``anthropic`` Python SDK.

    Requests opt in to Anthropic's server-side fallback, so a request that
    the model's safety classifiers decline is retried on the fallback model
    Anthropic recommends. Pass ``fallback=False`` to turn that off.
    """

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 fallback: bool = True, strict_tools: bool = True):
        """
        Args:
            api_key: API key. If omitted, the SDK finds credentials itself
                (``ANTHROPIC_API_KEY`` or a profile from ``ant auth login``).
            model: Model identifier
            fallback: Retry declined requests on Anthropic's recommended
                fallback model
            strict_tools: Hold tool arguments to their schema exactly
        """
        try:
            import anthropic
        except ImportError as error:
            raise BackendError("the anthropic package is not installed. "
                               "Install it with: pip install anthropic") from error
        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model or DEFAULT_MODELS["anthropic"]
        self.fallback = fallback
        self.strict_tools = strict_tools

    def _create(self, **request):
        sdk = self._sdk
        try:
            if self.fallback:
                response = self._client.beta.messages.create(
                    model=self.model, betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default", **request)
            else:
                response = self._client.messages.create(model=self.model, **request)
        except sdk.AuthenticationError as error:
            raise BackendError("Anthropic rejected the API key.") from error
        except sdk.RateLimitError as error:
            raise BackendError("Anthropic's rate limit was reached. Wait and try again.") from error
        except sdk.APIStatusError as error:
            raise BackendError(f"Anthropic returned an error ({error.status_code}): "
                               f"{error.message}") from error
        except sdk.APIConnectionError as error:
            raise BackendError("could not reach Anthropic. Check the internet "
                               "connection.") from error
        if response.stop_reason == "refusal":
            raise BackendError("the request was declined by the model's safeguards.")
        return response

    @staticmethod
    def _text(response) -> str:
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def generate(self, prompt, system=None, max_tokens=None, temperature=0.2) -> str:
        # Sampling parameters are not accepted by current Claude models, so
        # temperature is not sent.
        request: Dict[str, Any] = {"max_tokens": max_tokens or 16000,
                                   "messages": [{"role": "user", "content": prompt}]}
        if system:
            request["system"] = system
        return self._text(self._create(**request))

    def call_tools(self, prompt, tools, system=None, allow_none=True) -> ToolCall:
        tools = normalise_tools(tools)
        declared = []
        for tool in tools:
            schema = tool["parameters"]
            entry = {"name": tool["name"], "description": tool["description"],
                     "input_schema": _strict_schema(schema) if self.strict_tools else schema}
            if self.strict_tools:
                entry["strict"] = True
            declared.append(entry)
        # Forcing a tool is not accepted by current models: steer with the prompt.
        steer = ("Call one of the tools if any of them fits the request."
                 if allow_none else "You must answer by calling exactly one of the tools.")
        request: Dict[str, Any] = {
            "max_tokens": 16000, "tools": declared,
            "system": f"{system}\n\n{steer}" if system else steer,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self._create(**request)
        for block in response.content:
            if block.type == "tool_use":
                return ToolCall(block.name, dict(block.input))
        return ToolCall(None, {}, self._text(response))

    def __repr__(self) -> str:
        return f"AnthropicBackend(model='{self.model}')"


def api_key_from_environment(provider: str) -> Optional[str]:
    """The provider's key from its environment variable, if set."""
    for variable in KEY_VARIABLES.get(provider, ()):
        value = os.environ.get(variable)
        if value:
            return value
    return None


def make_api_backend(provider: str, api_key: Optional[str] = None,
                     model: Optional[str] = None, base_url: Optional[str] = None) -> Backend:
    """Build the backend for a provider."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}")
    extra = {"base_url": base_url} if base_url else {}
    if provider == "ollama":
        return OllamaBackend(model, **extra)
    if provider == "anthropic":
        return AnthropicBackend(api_key, model)
    if not api_key:
        variables = " or ".join(KEY_VARIABLES[provider])
        raise BackendError(f"no API key for {provider}. Set {variables}, or call "
                           f"actuneo.ai.configure(provider='{provider}', api_key=...).")
    if provider == "openai":
        return OpenAIBackend(api_key, model, **extra)
    return GeminiBackend(api_key, model, **extra)
