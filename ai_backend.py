"""AI backend abstraction for local Raspberry Pi and remote Jetson inference.

The local Raspberry Pi backend defaults to Ollama on the Pi CPU. The Raspberry Pi
AI HAT+ 26 TOPS (Hailo-8) is used by vision.py for accelerated perception; Hailo-8
cannot execute LLM/VLM workloads. The same interface can point at a Jetson Thor
Ollama server or an OpenAI-compatible server such as vLLM.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import httpx

try:
    import ollama
except Exception:  # pragma: no cover - runtime dependency on the target device
    ollama = None


class AIBackendError(RuntimeError):
    pass


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    try:
        return dict(value)
    except Exception:
        return {}




def _normalize_keep_alive(value: Any) -> Any:
    """Normalize Ollama keep_alive values across config/API versions.

    Ollama accepts numeric seconds (including negative numbers to keep a model
    resident) or duration strings such as ``5m``/``-1m``. A JSON string of
    ``"-1"`` is interpreted as a Go duration and fails because it has no unit.
    Older Be More configs used that exact string, so convert integer-looking
    strings to integers before sending them to Ollama.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                return value
    return value

def _normalize_spoken_model_name(value: str) -> str:
    value = value.lower().strip()
    replacements = {
        " point ": ".",
        " colon ": ":",
        " billion": "b",
        " b ": "b ",
    }
    padded = f" {value} "
    for old, new in replacements.items():
        padded = padded.replace(old, new)
    value = padded.strip()
    value = value.replace("qwen ", "qwen")
    value = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", value)
    value = re.sub(r"\s*:\s*", ":", value)
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


class AIBackendManager:
    """Routes text/VLM inference between configured local and server backends."""

    def __init__(self, config: Dict[str, Any], options: Optional[Dict[str, Any]] = None):
        self.config = copy.deepcopy(config or {})
        self.options = options or {}
        self.backends: Dict[str, Dict[str, Any]] = self.config.get("backends", {})
        self.default_backend = self.config.get("default_backend", "local")
        if self.default_backend not in self.backends and self.backends:
            self.default_backend = next(iter(self.backends))
        self.current_backend = self.default_backend
        self.fallback_to_local = bool(self.config.get("fallback_to_local", True))
        self._ollama_clients: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Backend/model state
    # ------------------------------------------------------------------
    def backend_config(self, backend: Optional[str] = None) -> Dict[str, Any]:
        name = backend or self.current_backend
        if name not in self.backends:
            raise AIBackendError(f"AI backend '{name}' is not configured")
        return self.backends[name]

    def get_model(self, vision: bool = False, backend: Optional[str] = None) -> str:
        cfg = self.backend_config(backend)
        key = "vision_model" if vision else "text_model"
        return str(cfg.get(key, "")).strip()

    def set_model(self, model: str, vision: bool = False, backend: Optional[str] = None) -> None:
        name = backend or self.current_backend
        cfg = self.backend_config(name)
        key = "vision_model" if vision else "text_model"
        cfg[key] = model

    def switch_backend(self, name_or_alias: str, *, verify: bool = True) -> str:
        """Switch backend only after the target is actually reachable.

        The previous implementation changed ``current_backend`` before proving
        that Thor could serve the configured model.  Because normal inference
        then silently fell back to local, the UI could say "thor" while the
        Raspberry Pi was doing the work.
        """
        normalized = name_or_alias.lower().strip()
        target = None
        for name, cfg in self.backends.items():
            aliases = [name] + list(cfg.get("aliases", []))
            if any(normalized == str(alias).lower().strip() for alias in aliases):
                target = name
                break
        if target is None:
            raise AIBackendError(f"Unknown AI backend: {name_or_alias}")

        previous = self.current_backend
        if verify:
            health = self.probe_backend(target, require_model=True)
            if not health.get("ok"):
                self.current_backend = previous
                raise AIBackendError(str(health.get("error") or f"{target} is unavailable"))

        self.current_backend = target
        return target

    def probe_backend(self, backend: Optional[str] = None, *, require_model: bool = True) -> Dict[str, Any]:
        """Verify endpoint reachability and, for Ollama, the configured model."""
        name = backend or self.current_backend
        cfg = self.backend_config(name)
        backend_type = str(cfg.get("type", "ollama")).lower()
        model = self.get_model(backend=name)
        try:
            if backend_type == "ollama":
                response = self._get_ollama_client(name).list()
                data = _as_dict(response)
                installed = []
                for item in data.get("models", []):
                    d = _as_dict(item)
                    model_name = d.get("model") or d.get("name")
                    if model_name:
                        installed.append(str(model_name))
                if require_model and model and model not in installed:
                    return {
                        "ok": False,
                        "backend": name,
                        "base_url": cfg.get("base_url"),
                        "model": model,
                        "models": installed,
                        "error": f"{name} is reachable but model {model} is not installed there",
                    }
                return {
                    "ok": True,
                    "backend": name,
                    "base_url": cfg.get("base_url"),
                    "model": model,
                    "models": installed,
                }

            if backend_type in {"openai", "openai_compatible", "vllm"}:
                base_url = cfg.get("base_url") or cfg.get("text_base_url")
                if not base_url:
                    return {"ok": False, "error": f"No base_url configured for {name}"}
                endpoint = str(base_url).rstrip("/") + "/models"
                response = requests.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {cfg.get('api_key', 'not-needed')}"},
                    timeout=max(2.0, float(cfg.get("connect_timeout_seconds", 4.0))),
                )
                response.raise_for_status()
                installed = [str(x.get("id")) for x in response.json().get("data", []) if x.get("id")]
                if require_model and model and installed and model not in installed:
                    return {"ok": False, "error": f"{name} is reachable but model {model} is not served there"}
                return {"ok": True, "backend": name, "base_url": base_url, "model": model, "models": installed}

            return {"ok": False, "error": f"Unsupported backend type {backend_type!r}"}
        except Exception as exc:
            return {
                "ok": False,
                "backend": name,
                "base_url": cfg.get("base_url"),
                "model": model,
                "error": f"cannot reach {name} at {cfg.get('base_url')}: {exc}",
            }

    def status_text(self) -> str:
        cfg = self.backend_config()
        return (
            f"Using {self.current_backend} AI with text model {cfg.get('text_model')} "
            f"and vision model {cfg.get('vision_model')}."
        )

    # ------------------------------------------------------------------
    # Voice model/backend switching
    # ------------------------------------------------------------------
    def parse_speech_command(self, text: str) -> Optional[str]:
        """Handle explicit spoken backend/model commands.

        Returns a user-facing confirmation string when handled, otherwise None.
        Commands are intentionally conservative to avoid interpreting questions like
        "should I switch to the server?" as state changes.
        """
        raw = text.strip()
        lower = raw.lower().strip(" .?!")

        if re.fullmatch(r"(?:what|which) (?:ai )?(?:backend|model)(?: are you using| is active| are you on)?", lower):
            return self.status_text()

        imperative = re.match(r"^(switch|change|use|go|move)\b", lower)
        if not imperative:
            return None

        # Backend switch first.  Do not confirm until the target endpoint and
        # configured model are actually reachable.
        for name, cfg in self.backends.items():
            aliases = [name] + list(cfg.get("aliases", []))
            for alias in sorted(aliases, key=lambda x: len(str(x)), reverse=True):
                alias_l = str(alias).lower().strip()
                if re.search(rf"\b{re.escape(alias_l)}\b", lower):
                    try:
                        self.switch_backend(name, verify=True)
                        return self.status_text()
                    except AIBackendError as exc:
                        return f"I could not switch to {name}. {exc}. I am still using {self.current_backend}."

        # Model switch on the current backend.
        cfg = self.backend_config()
        is_vision = "vision" in lower or "vlm" in lower
        available_key = "vision_models" if is_vision else "text_models"
        configured_models: List[str] = list(cfg.get(available_key, []))
        current_model = self.get_model(vision=is_vision)
        if current_model and current_model not in configured_models:
            configured_models.append(current_model)

        lower_norm = _normalize_spoken_model_name(lower)
        for model in configured_models:
            model_norm = _normalize_spoken_model_name(model)
            if model_norm and model_norm in lower_norm:
                # For Ollama, verify the model is actually present when possible.
                installed = self.list_models(self.current_backend)
                if installed and model not in installed:
                    return (
                        f"{model} is configured but is not installed on {self.current_backend}. "
                        f"Install or pull it first."
                    )
                self.set_model(model, vision=is_vision)
                kind = "vision" if is_vision else "text"
                return f"Switched the {kind} model to {model} on {self.current_backend}."

        return None

    def cancel_pending_requests(self) -> None:
        """Close live HTTP clients so a manual recovery can abort stuck I/O.

        New clients are created lazily on the next request. This is especially
        useful when a Thor server disappears while an Ollama request is active.
        """
        clients = list(self._ollama_clients.values())
        self._ollama_clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------
    def warmup(self) -> None:
        cfg = self.backend_config()
        if cfg.get("type", "ollama") == "ollama":
            client = self._get_ollama_client(self.current_backend)
            model = self.get_model()
            if model:
                try:
                    client.generate(model=model, prompt="", keep_alive=-1)
                except Exception as exc:
                    raise AIBackendError(str(exc)) from exc

    def unload(self, backend: Optional[str] = "local") -> None:
        """Unload Ollama models without unexpectedly evicting a shared Thor model.

        By default only the Pi-local backend is unloaded when this client exits.
        Pass ``backend=None`` explicitly if every configured Ollama backend should
        be unloaded.
        """
        targets = list(self.backends) if backend is None else [backend]
        for name in targets:
            cfg = self.backends.get(name, {})
            if cfg.get("type", "ollama") != "ollama":
                continue
            model = str(cfg.get("text_model", "")).strip()
            if not model:
                continue
            try:
                self._get_ollama_client(name).generate(model=model, prompt="", keep_alive=0)
            except Exception:
                pass

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        vision: bool = False,
        image_path: Optional[str] = None,
        backend: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> Dict[str, Any]:
        target = backend or self.current_backend
        model = self.get_model(vision=vision, backend=target)
        try:
            response = self._chat_backend(target, messages, tools=tools, vision=vision, image_path=image_path)
            response["requested_backend"] = target
            response["actual_backend"] = target
            response["fallback_used"] = False
            response["model"] = model
            return response
        except Exception as exc:
            if (
                allow_fallback
                and self.fallback_to_local
                and target != "local"
                and "local" in self.backends
            ):
                print(f"[AI] {target} failed ({exc}); falling back to local and changing active state.", flush=True)
                local_model = self.get_model(vision=vision, backend="local")
                try:
                    response = self._chat_backend("local", messages, tools=tools, vision=vision, image_path=image_path)
                except Exception as local_exc:
                    # Surface both failures as one clean error rather than
                    # letting the fallback's exception escape unwrapped.
                    raise AIBackendError(
                        f"{target} failed ({exc}) and the local fallback also failed ({local_exc})"
                    ) from local_exc
                # Keep the UI/status truthful after a remote failure.  The user
                # can explicitly switch back to Thor once its health check passes.
                self.current_backend = "local"
                response["requested_backend"] = target
                response["actual_backend"] = "local"
                response["fallback_used"] = True
                response["fallback_from"] = target
                response["fallback_reason"] = str(exc)
                response["model"] = local_model
                return response
            if isinstance(exc, AIBackendError):
                raise
            raise AIBackendError(str(exc)) from exc

    def vision_describe(
        self,
        prompt: str,
        image_path: str,
        *,
        perception_context: str = "",
        backend: Optional[str] = None,
    ) -> str:
        content = prompt.strip() or "Describe what you see clearly and briefly."
        if perception_context:
            content += (
                "\n\nA Hailo object detector also reported the following scene hints. "
                "Use them as hints, but trust the image if they conflict:\n" + perception_context
            )
        response = self.chat(
            [{"role": "user", "content": content}],
            vision=True,
            image_path=image_path,
            backend=backend,
            tools=None,
        )
        return str(response.get("content", "")).strip()

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _chat_backend(
        self,
        backend: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        vision: bool,
        image_path: Optional[str],
    ) -> Dict[str, Any]:
        cfg = self.backend_config(backend)
        backend_type = str(cfg.get("type", "ollama")).lower()
        model = self.get_model(vision=vision, backend=backend)
        if not model:
            raise AIBackendError(f"No {'vision' if vision else 'text'} model configured for {backend}")

        if backend_type == "ollama":
            return self._chat_ollama(backend, model, messages, tools, image_path)
        if backend_type in {"openai", "openai_compatible", "vllm"}:
            return self._chat_openai_compatible(backend, model, messages, tools, image_path, vision)
        raise AIBackendError(f"Unsupported backend type '{backend_type}'")

    def _get_ollama_client(self, backend: str):
        if ollama is None:
            raise AIBackendError("The 'ollama' Python package is not installed")
        if backend not in self._ollama_clients:
            cfg = self.backend_config(backend)
            host = cfg.get("base_url", "http://127.0.0.1:11434")
            # ollama-python intentionally defaults to timeout=None. That is fine
            # for a CLI, but a voice robot can otherwise become permanently
            # unresponsive when a remote Thor disappears mid-request.
            request_timeout = max(5.0, float(cfg.get("timeout_seconds", 60.0)))
            connect_timeout = max(1.0, float(cfg.get("connect_timeout_seconds", 4.0)))
            timeout = httpx.Timeout(request_timeout, connect=connect_timeout)
            self._ollama_clients[backend] = ollama.Client(host=host, timeout=timeout)
        return self._ollama_clients[backend]

    def _chat_ollama(
        self,
        backend: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        image_path: Optional[str],
    ) -> Dict[str, Any]:
        client = self._get_ollama_client(backend)
        payload_messages = copy.deepcopy(messages)
        for message in payload_messages:
            if message.get("role") == "tool":
                message.pop("tool_call_id", None)
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        fn["arguments"] = {}
        if image_path:
            # Attach the image to the last user message.
            for message in reversed(payload_messages):
                if message.get("role") == "user":
                    message["images"] = [image_path]
                    break

        cfg = self.backend_config(backend)
        options = copy.deepcopy(self.options)
        backend_options = cfg.get("options", {})
        if isinstance(backend_options, dict):
            options.update(backend_options)

        # Do not force a per-request context size on remote Ollama servers by
        # default. Ollama may unload/reload an already-resident runner when
        # num_ctx differs from the server/model context, which is especially
        # disruptive for large Thor models. Let Thor own OLLAMA_CONTEXT_LENGTH
        # unless explicitly opted back in with send_num_ctx_per_request=true.
        send_num_ctx = bool(cfg.get("send_num_ctx_per_request", backend == "local"))
        if not send_num_ctx and "num_ctx" in options:
            inherited_ctx = options.pop("num_ctx", None)
            print(
                f"[AI] {backend}: inheriting Ollama server context "
                f"(not sending per-request num_ctx={inherited_ctx}).",
                flush=True,
            )

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "stream": False,
            "options": options,
            # Qwen thinking models spend substantial CPU time generating an
            # internal reasoning trace. A voice assistant should optimize for
            # response latency; reasoning can be re-enabled per backend in config.
            "think": cfg.get("think", False),
        }
        if tools:
            kwargs["tools"] = tools
        keep_alive = _normalize_keep_alive(cfg.get("keep_alive", -1))
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive

        # `think` is only accepted by newer ollama-python releases. On an older
        # client every request otherwise died with an unexpected-keyword
        # TypeError, which surfaced as "my AI backend is unavailable" for the
        # entire session. Drop unsupported kwargs and retry instead of failing.
        response = None
        for _attempt in range(3):
            try:
                response = client.chat(**kwargs)
                break
            except TypeError as exc:
                dropped = None
                for candidate in ("think", "keep_alive", "options"):
                    if candidate in kwargs and candidate in str(exc):
                        dropped = candidate
                        break
                if dropped is None:
                    raise
                print(
                    f"[AI] Installed ollama client rejected '{dropped}'; retrying without it.",
                    flush=True,
                )
                kwargs.pop(dropped, None)
        if response is None:
            raise AIBackendError("the installed ollama client rejected the request parameters")

        response_dict = _as_dict(response)
        message = _as_dict(response_dict.get("message", getattr(response, "message", None)))
        normalized = self._normalize_message(message)

        # Keep lightweight timing data so the host can distinguish model load,
        # prompt evaluation, and token generation delays on constrained hardware.
        normalized["metrics"] = {
            key: response_dict.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if response_dict.get(key) is not None
        }
        return normalized

    def _chat_openai_compatible(
        self,
        backend: str,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        image_path: Optional[str],
        vision: bool,
    ) -> Dict[str, Any]:
        cfg = self.backend_config(backend)
        base_url = cfg.get("vision_base_url" if vision else "text_base_url") or cfg.get("base_url")
        if not base_url:
            raise AIBackendError(f"No base_url configured for {backend}")
        endpoint = str(base_url).rstrip("/") + "/chat/completions"
        api_key = cfg.get("api_key", "not-needed")

        payload_messages = copy.deepcopy(messages)
        for message in payload_messages:
            if message.get("role") == "tool":
                message.pop("tool_name", None)
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args)
        if image_path:
            mime = "image/jpeg" if str(image_path).lower().endswith((".jpg", ".jpeg")) else "image/png"
            encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            for message in reversed(payload_messages):
                if message.get("role") == "user":
                    text_content = message.get("content", "")
                    message["content"] = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ]
                    break

        body: Dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "temperature": self.options.get("temperature", 0.7),
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        timeout = float(cfg.get("timeout_seconds", 180))
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
        if response.status_code >= 400:
            raise AIBackendError(f"{backend} returned HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise AIBackendError(f"{backend} returned no choices")
        return self._normalize_message(choices[0].get("message", {}))

    @staticmethod
    def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
        content = message.get("content") or ""
        tool_calls_out: List[Dict[str, Any]] = []
        for call in message.get("tool_calls") or []:
            call_dict = _as_dict(call)
            fn = _as_dict(call_dict.get("function"))
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tool_calls_out.append(
                {
                    "id": call_dict.get("id"),
                    "type": call_dict.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args if isinstance(args, dict) else {},
                    },
                }
            )
        return {"content": str(content), "tool_calls": tool_calls_out, "raw": message}

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------
    def list_models(self, backend: Optional[str] = None) -> List[str]:
        name = backend or self.current_backend
        cfg = self.backend_config(name)
        backend_type = str(cfg.get("type", "ollama")).lower()
        try:
            if backend_type == "ollama":
                response = self._get_ollama_client(name).list()
                data = _as_dict(response)
                result = []
                for item in data.get("models", []):
                    d = _as_dict(item)
                    model_name = d.get("model") or d.get("name")
                    if model_name:
                        result.append(str(model_name))
                return result
            if backend_type in {"openai", "openai_compatible", "vllm"}:
                base_url = cfg.get("base_url") or cfg.get("text_base_url")
                if not base_url:
                    return []
                endpoint = str(base_url).rstrip("/") + "/models"
                response = requests.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {cfg.get('api_key', 'not-needed')}"},
                    timeout=5,
                )
                response.raise_for_status()
                return [str(x.get("id")) for x in response.json().get("data", []) if x.get("id")]
        except Exception:
            return []
        return []
