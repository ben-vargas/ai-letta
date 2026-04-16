"""OpenAI native compaction via POST /v1/responses/compact."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import httpx

from letta.llm_api.openai_client import use_responses_api
from letta.log import get_logger
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import (
    ImageContent,
    OpenAICompactionContent,
    SummarizedReasoningContent,
    SummarizedReasoningContentPart,
    TextContent,
    UrlImage,
)
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, OpenAIFunction, OpenAIToolCall, ToolReturn
from letta.schemas.user import User

logger = get_logger(__name__)

COMPACT_TIMEOUT_SECONDS = 120
COMPACTION_SUMMARY_TEXT = "Context compacted via OpenAI /v1/responses/compact."


async def _resolve_api_key(actor: User, llm_config: LLMConfig) -> str:
    """Resolve the OpenAI API key, checking BYOK overrides first."""
    try:
        from letta.llm_api.openai_client import OpenAIClient

        client = OpenAIClient(actor=actor)
        api_key, _, _ = await client.get_byok_overrides_async(llm_config)
        if api_key:
            return api_key
    except Exception as exc:
        logger.debug(f"Failed to resolve OpenAI BYOK override for compact mode: {type(exc).__name__}: {exc}")

    try:
        from letta.settings import model_settings

        if model_settings.openai_api_key:
            return model_settings.openai_api_key
    except Exception as exc:
        logger.debug(f"Failed to read configured OpenAI API key for compact mode: {type(exc).__name__}: {exc}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    raise ValueError("No OpenAI API key found for openai_compact mode. Set OPENAI_API_KEY or configure BYOK.")


def _normalize_compact_url(base_url: Optional[str]) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/responses/compact", parsed.query, parsed.fragment))


def _build_tools_for_compact(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
            **({"strict": tool["strict"]} if "strict" in tool else {}),
        }
        for tool in tools
    ]


def _content_parts_to_text(parts: object) -> str:
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""

    text_parts: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "summary_text"):
            text_parts.append(part.get("text", ""))
        elif part_type == "refusal":
            text_parts.append(part.get("refusal", ""))
    return "\n".join(part for part in text_parts if part)


def _tool_output_to_content(output: object) -> object:
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return ""

    content_parts = []
    for part in output:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "input_text":
            content_parts.append(TextContent(text=part.get("text", "")))
        elif part_type == "input_image":
            image_url = part.get("image_url")
            if image_url:
                content_parts.append(ImageContent(source=UrlImage(url=image_url)))
        elif part_type == "input_file":
            filename = part.get("filename") or part.get("file_id") or part.get("file_url") or "file"
            content_parts.append(TextContent(text=f"[File output: {filename}]"))

    return content_parts if content_parts else ""


def _message_item_to_message(
    item: dict,
    agent_id: str,
    run_id: Optional[str],
    step_id: Optional[str],
) -> Optional[Message]:
    role = item.get("role")
    if role == "developer":
        return None

    text = _content_parts_to_text(item.get("content", ""))
    openai_phase = item.get("phase") if role == "assistant" else None
    content = [TextContent(text=text, openai_phase=openai_phase)] if text else []

    if role == "user":
        return Message(role=MessageRole.user, content=content or [TextContent(text="")], agent_id=agent_id, run_id=run_id, step_id=step_id)
    if role == "assistant":
        return Message(role=MessageRole.assistant, content=content, agent_id=agent_id, run_id=run_id, step_id=step_id)
    return None


def _reasoning_item_to_message(
    item: dict,
    agent_id: str,
    run_id: Optional[str],
    step_id: Optional[str],
) -> Optional[Message]:
    reasoning_id = item.get("id")
    summary = item.get("summary")
    if not reasoning_id or not isinstance(summary, list):
        return None

    summary_parts = []
    for idx, part in enumerate(summary):
        if isinstance(part, dict) and part.get("type") == "summary_text":
            summary_parts.append(SummarizedReasoningContentPart(index=idx, text=part.get("text", "")))

    reasoning_content = SummarizedReasoningContent(
        id=reasoning_id,
        summary=summary_parts,
        encrypted_content=item.get("encrypted_content"),
    )
    return Message(
        role=MessageRole.assistant,
        content=[reasoning_content],
        agent_id=agent_id,
        run_id=run_id,
        step_id=step_id,
    )


def _process_compact_output(
    output_items: List[dict],
    agent_id: str,
    system_message: Message,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
) -> Tuple[str, List[Message]]:
    """Convert compact output to Letta messages while preserving retained items."""
    compacted_messages: List[Message] = [system_message]
    saw_compaction = False

    for item in output_items:
        item_type = item.get("type")

        if item_type == "compaction":
            saw_compaction = True
            compacted_messages.append(
                Message(
                    role=MessageRole.summary,
                    content=[
                        TextContent(text=COMPACTION_SUMMARY_TEXT),
                        OpenAICompactionContent(
                            encrypted_content=item.get("encrypted_content", ""),
                            compaction_id=item.get("id"),
                        )
                    ],
                    agent_id=agent_id,
                    run_id=run_id,
                    step_id=step_id,
                )
            )
            continue

        if item_type == "function_call":
            compacted_messages.append(
                Message(
                    role=MessageRole.assistant,
                    content=[],
                    tool_calls=[
                        OpenAIToolCall(
                            id=item.get("call_id", ""),
                            type="function",
                            function=OpenAIFunction(
                                name=item.get("name", ""),
                                arguments=item.get("arguments", "{}"),
                            ),
                        )
                    ],
                    agent_id=agent_id,
                    run_id=run_id,
                    step_id=step_id,
                )
            )
            continue

        if item_type == "function_call_output":
            compacted_messages.append(
                Message(
                    role=MessageRole.tool,
                    tool_returns=[
                        ToolReturn(
                            tool_call_id=item.get("call_id"),
                            status="success",
                            func_response=_tool_output_to_content(item.get("output")),
                        )
                    ],
                    agent_id=agent_id,
                    run_id=run_id,
                    step_id=step_id,
                )
            )
            continue

        if item_type == "reasoning":
            reasoning_message = _reasoning_item_to_message(item, agent_id=agent_id, run_id=run_id, step_id=step_id)
            if reasoning_message is not None:
                compacted_messages.append(reasoning_message)
            continue

        message = _message_item_to_message(item, agent_id=agent_id, run_id=run_id, step_id=step_id)
        if message is not None:
            compacted_messages.append(message)
            continue

        logger.debug(f"Skipping unsupported compact output item: {item_type or item.get('role', 'unknown')}")

    if not saw_compaction:
        raise ValueError("OpenAI compact response did not include a compaction item.")

    return COMPACTION_SUMMARY_TEXT, compacted_messages


async def openai_compact(
    actor: User,
    agent_id: str,
    agent_llm_config: LLMConfig,
    messages: List[Message],
    tools: Optional[List[dict]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
) -> Tuple[str, List[Message]]:
    """Compact a Letta transcript using OpenAI's standalone compact endpoint."""
    if agent_llm_config.model_endpoint_type != "openai":
        raise ValueError("openai_compact mode is only supported for OpenAI-backed models.")
    if not use_responses_api(agent_llm_config):
        raise ValueError("openai_compact mode requires subsequent turns to use the OpenAI Responses API.")
    if not messages or messages[0].role != MessageRole.system:
        raise ValueError("Expected the system message as the first compaction input item.")

    api_key = await _resolve_api_key(actor, agent_llm_config)
    compact_url = _normalize_compact_url(agent_llm_config.model_endpoint)

    payload = {
        "model": agent_llm_config.model or "gpt-5.4",
        "input": Message.to_openai_responses_dicts_from_list(messages),
    }
    compact_tools = _build_tools_for_compact(tools)
    if compact_tools:
        payload["tools"] = compact_tools

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info(
        f"Calling OpenAI compact endpoint: model={payload['model']}, input_items={len(payload['input'])}, "
        f"tools={len(compact_tools) if compact_tools else 0}"
    )

    async with httpx.AsyncClient(timeout=COMPACT_TIMEOUT_SECONDS) as client:
        response = await client.post(compact_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

    output_items = result.get("output", [])
    usage = result.get("usage", {})

    logger.info(
        f"OpenAI compact response: output_items={len(output_items)}, "
        f"input_tokens={usage.get('input_tokens', '?')}, output_tokens={usage.get('output_tokens', '?')}"
    )

    summary_text, compacted_messages = _process_compact_output(
        output_items=output_items,
        agent_id=agent_id,
        system_message=messages[0],
        run_id=run_id,
        step_id=step_id,
    )
    return summary_text, compacted_messages
