"""Unit tests for OpenAI native compaction via /v1/responses/compact."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from letta.agents.letta_agent_v3 import extract_compaction_stats_from_message
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import OpenAICompactionContent, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.user import User
from letta.services.summarizer.openai_compact import (
    _normalize_compact_url,
    _process_compact_output,
    openai_compact,
)
from letta.services.summarizer.summarizer_config import CompactionSettings
from letta.system import package_summarize_message_no_counts


def _make_user() -> User:
    return User(id="user-test-123", name="test_user")


def _make_llm_config(model: str = "gpt-5.4", endpoint: str = "https://api.openai.com/v1") -> LLMConfig:
    return LLMConfig(
        model=model,
        model_endpoint_type="openai",
        model_endpoint=endpoint,
        context_window=272000,
    )


def _make_system_message(agent_id: str = "agent-1") -> Message:
    return Message(
        role=MessageRole.system,
        content=[TextContent(text="You are a helpful coding assistant.")],
        agent_id=agent_id,
    )


def test_compaction_settings_accepts_openai_compact_mode():
    settings = CompactionSettings(mode="openai_compact")
    assert settings.mode == "openai_compact"


def test_normalize_compact_url_appends_v1():
    assert _normalize_compact_url("https://api.openai.com") == "https://api.openai.com/v1/responses/compact"
    assert _normalize_compact_url("https://api.openai.com/v1") == "https://api.openai.com/v1/responses/compact"


def test_process_compact_output_preserves_retained_tool_items():
    system_msg = _make_system_message()
    output_items = [
        {"type": "compaction", "id": "cmp-1", "encrypted_content": "enc-data"},
        {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "Working..."}]},
        {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": '{"query":"python"}'},
        {"type": "function_call_output", "call_id": "call-1", "output": "tool result"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "follow-up"}]},
    ]

    summary_text, messages = _process_compact_output(
        output_items=output_items,
        agent_id="agent-1",
        system_message=system_msg,
    )

    assert summary_text
    assert len(messages) == 6
    assert messages[1].role == MessageRole.summary
    assert any(isinstance(content, OpenAICompactionContent) for content in messages[1].content)
    assert messages[2].role == MessageRole.assistant
    assert messages[2].content[0].openai_phase == "commentary"
    assert messages[3].role == MessageRole.assistant
    assert messages[3].tool_calls[0].id == "call-1"
    assert messages[4].role == MessageRole.tool
    assert messages[4].tool_returns[0].tool_call_id == "call-1"
    assert messages[5].role == MessageRole.user


def test_openai_compaction_summary_serializes_as_compaction_item():
    packed_summary = package_summarize_message_no_counts(
        summary="Context compacted via OpenAI /v1/responses/compact.",
        timezone="UTC",
        compaction_stats={
            "trigger": "post_step_context_check",
            "context_tokens_before": 100,
            "context_tokens_after": 40,
            "context_window": 1000,
            "messages_count_before": 10,
            "messages_count_after": 4,
        },
        mode="openai_compact",
    )
    msg = Message(
        role=MessageRole.summary,
        content=[
            TextContent(text=packed_summary),
            OpenAICompactionContent(
                encrypted_content="test-encrypted",
                compaction_id="cmp-456",
            ),
        ],
        agent_id="agent-1",
    )

    dicts = msg.to_openai_responses_dicts()
    assert len(dicts) == 1
    assert dicts[0]["type"] == "compaction"
    assert dicts[0]["encrypted_content"] == "test-encrypted"
    assert dicts[0]["id"] == "cmp-456"

    summary_msg = msg._convert_summary_message(as_user_message=False)
    assert "compacted" in summary_msg.summary.lower()
    assert summary_msg.compaction_stats is not None
    assert summary_msg.compaction_stats.context_tokens_before == 100

    extracted = extract_compaction_stats_from_message(msg)
    assert extracted is not None
    assert extracted.messages_count_after == 4


def test_extract_compaction_stats_from_message_uses_summary_metadata_fallback():
    msg = Message(
        role=MessageRole.summary,
        content=[
            OpenAICompactionContent(
                encrypted_content="test-encrypted",
                compaction_id="cmp-789",
            )
        ],
        name=json.dumps(
            {
                "compaction_stats": {
                    "trigger": "post_step_context_check",
                    "context_tokens_before": 100,
                    "context_tokens_after": 40,
                    "context_window": 1000,
                    "messages_count_before": 10,
                    "messages_count_after": 4,
                }
            }
        ),
        agent_id="agent-1",
    )

    extracted = extract_compaction_stats_from_message(msg)
    assert extracted is not None
    assert extracted.trigger == "post_step_context_check"
    assert extracted.messages_count_after == 4


def test_assistant_text_content_round_trips_openai_phase_in_responses():
    msg = Message(
        role=MessageRole.assistant,
        content=[TextContent(text="Still working on it", openai_phase="commentary")],
        agent_id="agent-1",
    )

    dicts = msg.to_openai_responses_dicts()
    assert len(dicts) == 1
    assert dicts[0]["role"] == "assistant"
    assert dicts[0]["content"] == "Still working on it"
    assert dicts[0]["phase"] == "commentary"


@pytest.mark.asyncio
async def test_openai_compact_rejects_non_responses_models():
    actor = _make_user()
    llm_config = _make_llm_config(model="gpt-4.1")
    with pytest.raises(ValueError, match="Responses API"):
        await openai_compact(
            actor=actor,
            agent_id="agent-1",
            agent_llm_config=llm_config,
            messages=[_make_system_message()],
        )


@pytest.mark.asyncio
async def test_openai_compact_calls_normalized_endpoint():
    actor = _make_user()
    llm_config = _make_llm_config(endpoint="https://api.openai.com")
    messages = [
        _make_system_message("agent-test-1"),
        Message(role=MessageRole.user, content=[TextContent(text="Tell me about Python")], agent_id="agent-test-1"),
    ]

    mock_response_data = {
        "id": "resp-compact-1",
        "output": [{"type": "compaction", "id": "cmp-1", "encrypted_content": "mock-encrypted-content"}],
        "usage": {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
    }
    mock_http_response = httpx.Response(
        status_code=200,
        json=mock_response_data,
        request=httpx.Request("POST", "https://api.openai.com/v1/responses/compact"),
    )

    with patch(
        "letta.services.summarizer.openai_compact._resolve_api_key",
        new_callable=AsyncMock,
        return_value="test-key-123",
    ):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_http_response) as mock_post:
            summary_text, compacted_messages = await openai_compact(
                actor=actor,
                agent_id="agent-test-1",
                agent_llm_config=llm_config,
                messages=messages,
            )

    assert summary_text
    assert compacted_messages[1].role == MessageRole.summary
    called_url = mock_post.call_args.args[0]
    assert called_url == "https://api.openai.com/v1/responses/compact"
