"""The LLM behind an interface, and one implementation of it.

Why an interface at all
-----------------------
So the traffic domain never learns what a chat completion is. Everything above
this module deals in facts and an advisory; swapping OpenRouter for Anthropic,
Gemini, Groq or a local model is a new class here and no change anywhere else.

Why OpenRouter
--------------
Because the project already configured it. ``OPENROUTER_API_KEY``,
``OPENROUTER_MODEL``, base URL, timeout and token budget have been in
:class:`~app.core.config.Settings` since Phase 1 in anticipation of Phase 12, and
``/api/health`` already reports whether credentials are present. Adding a second
provider's settings beside them would have created two half-configured paths.

Why stdlib HTTP
---------------
One JSON POST does not justify a new runtime dependency. ``urllib.request``
handles it, with an explicit timeout, and the call sits in a synchronous route
handler that FastAPI runs in a worker thread -- the same pattern as every other
handler in this service.

Failure is a return value, not an exception at the top
-------------------------------------------------------
Every provider failure -- no key, timeout, HTTP error, unparseable body -- is
raised as :class:`AdvisorProviderError` and caught by the service, which falls
back to a deterministic advisory. A traffic dashboard that shows an error where
an advisory should be is less useful than one that says what it can from the
facts and admits no model was involved.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol

from app.core.config import Settings
from app.core.errors import TrafficSystemError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Sent so OpenRouter attributes traffic from this project. No user data.
_REFERER = "https://github.com/genai-traffic-management"
_TITLE = "GenAI Traffic Management - Traffic Advisor"


class AdvisorProviderError(TrafficSystemError):
    """The language model could not be reached, or did not answer usably.

    A 502 like any other upstream failure. It is rarely surfaced: the service
    catches it and answers with a deterministic advisory instead, because the
    operator's question ("what is happening at this junction?") still has an
    answer that does not need a model.
    """

    status_code = 502
    error_code = "advisor_provider_error"
    is_expected = True


class TrafficAdvisorLLM(Protocol):
    """A language model that can be asked for one advisory.

    Deliberately narrow: two prompts in, raw text out. Parsing and validating
    that text is the service's job, not the provider's, so a provider swap
    cannot quietly change what counts as a valid advisory.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the model, recorded on every advisory it produces."""
        ...

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's raw reply, or raise :class:`AdvisorProviderError`."""
        ...


class OpenRouterAdvisorLLM:
    """OpenRouter chat-completions implementation of :class:`TrafficAdvisorLLM`."""

    __slots__ = ("_api_key", "_base_url", "_max_tokens", "_model", "_timeout_s")

    def __init__(self, settings: Settings) -> None:
        key = settings.openrouter_api_key
        if key is None or not settings.openrouter_model:
            msg = (
                "OpenRouter is not configured: set OPENROUTER_API_KEY and "
                "OPENROUTER_MODEL to enable model-written advisories."
            )
            raise AdvisorProviderError(msg)
        # Unwrapped here and never stored anywhere it could be serialised. It
        # lives on this object, goes into one header, and is not part of any
        # response model in this service.
        self._api_key = key.get_secret_value()
        self._model = settings.openrouter_model
        self._base_url = settings.openrouter_base_url.rstrip("/")
        self._timeout_s = settings.openrouter_timeout_s
        self._max_tokens = settings.openrouter_max_tokens

    @property
    def model_id(self) -> str:
        """The configured model identifier."""
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """POST one chat completion and return the assistant's message text."""
        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # Lowest sensible temperature: this is an explanation of fixed
                # facts, and there is nothing here worth being creative about.
                "temperature": 0.1,
                "max_tokens": self._max_tokens,
                # Honoured by providers that support it and ignored by the rest,
                # which is why the reply is parsed defensively regardless.
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")

        request = urllib.request.Request(  # noqa: S310 - fixed https base URL
            f"{self._base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": _REFERER,
                "X-Title": _TITLE,
            },
        )

        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed https base URL
                request, timeout=self._timeout_s
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            msg = (
                f"The advisor model did not respond within {self._timeout_s:g}s."
            )
            raise AdvisorProviderError(msg, model=self._model) from exc
        except urllib.error.HTTPError as exc:
            # The provider's body may carry the reason (bad key, no credit, model
            # not found) but may also echo the request. Only the status is kept.
            msg = f"The advisor model returned HTTP {exc.code}."
            raise AdvisorProviderError(msg, model=self._model, status=exc.code) from exc
        except (urllib.error.URLError, OSError) as exc:
            msg = "The advisor model could not be reached."
            raise AdvisorProviderError(msg, model=self._model) from exc
        except json.JSONDecodeError as exc:
            msg = "The advisor model returned a body that is not JSON."
            raise AdvisorProviderError(msg, model=self._model) from exc

        return _extract_message(payload, self._model)


def _extract_message(payload: Any, model: str) -> str:
    """Pull the assistant text out of a chat-completions body.

    Defensive rather than trusting: a provider that changes its envelope, or
    returns an error object with a 200, must produce a clean provider error
    here instead of a ``TypeError`` three frames later.
    """
    if not isinstance(payload, dict):
        msg = "The advisor model returned an unexpected response shape."
        raise AdvisorProviderError(msg, model=model)

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "The advisor model returned no completion."
        raise AdvisorProviderError(msg, model=model)

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        msg = "The advisor model returned an empty completion."
        raise AdvisorProviderError(msg, model=model)
    return content


def build_provider(settings: Settings) -> TrafficAdvisorLLM | None:
    """The configured provider, or ``None`` when none is configured.

    ``None`` is a working state, not a failure: the service answers from the
    facts and labels the result as deterministic. Raising here would make an
    unconfigured demo look broken when it is merely offline.
    """
    if not settings.genai_configured:
        return None
    try:
        return OpenRouterAdvisorLLM(settings)
    except AdvisorProviderError:  # pragma: no cover - guarded by genai_configured
        logger.warning("advisor provider configured but could not be constructed")
        return None
