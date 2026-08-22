"""Pulling a JSON object out of a local LLM's response.

Small local models routinely ignore "respond with only JSON" and wrap the
object in prose or a code fence, so every JSON-mode prompt in this package
parses through here rather than calling `json.loads` on the raw response.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

#: Matches the first top-level {...} block, tolerating prose or a code fence
#: the model wrapped the JSON in despite being asked not to. Greedy on
#: purpose: the closing brace of the *last* object is the right end for a
#: nested payload, and trailing prose after it is rare.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(response: str | None) -> dict | None:
    """Return the first JSON object in `response`, or None if there isn't one.

    A non-object payload (a bare list, a string) is treated as absent: every
    prompt in this package asks for an object, so anything else is the model
    having ignored the schema.
    """
    if not response:
        return None

    match = _JSON_BLOCK.search(response)
    payload = match.group(0) if match else response
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        log.warning("LLM returned unparsable JSON: %.200s", response)
        return None

    return data if isinstance(data, dict) else None
