"""Messages turn coalescing precedes tool-call boundary validation."""

import copy

import pytest
from deepseek_recipe import (
    ConversionError,
    ConversionOptions,
    DeepseekV41Encoding,
    MessagesRequest,
)

USES = [
    {"type": "tool_use", "id": ident, "name": "lookup", "input": {"key": key}}
    for ident, key in (("call_a", "a"), ("call_b", "b"))
]
RESULTS = [
    {"type": "tool_result", "tool_use_id": ident, "content": value}
    for ident, value in (("call_a", "1"), ("call_b", "2"))
]


def convert(messages):
    return MessagesRequest(
        {
            "model": "DeepSeek-V4.1-Flash",
            "thinking": {"type": "disabled"},
            "messages": copy.deepcopy(messages),
            "tools": [
                {"name": "lookup", "input_schema": {"type": "object", "properties": {}}}
            ],
        }
    ).convert(ConversionOptions())


def render(messages):
    return (
        DeepseekV41Encoding().render_conversation(convert(messages).conversation).prompt
    )


def history():
    return [
        {"role": "user", "content": "Look up both keys."},
        {"role": "assistant", "content": USES},
    ]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("split_assistant", [False, True])
def test_parallel_tool_results_across_consecutive_user_messages(
    reverse, split_assistant
):
    results = RESULTS[::-1] if reverse else RESULTS
    messages = history()
    if split_assistant:
        messages[1:] = [{"role": "assistant", "content": [use]} for use in USES]
    split = messages + [{"role": "user", "content": [result]} for result in results]
    combined = history() + [{"role": "user", "content": results}]
    assert render(split) == render(combined)
    converted = convert(split).conversation.messages
    assert [message.role for message in converted] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message.tool_call_id for message in converted[-2:]] == [
        result["tool_use_id"] for result in results
    ]


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("raw_first", [False, True])
@pytest.mark.parametrize("raw_second", [False, True])
def test_raw_and_block_content_coalesce(role, raw_first, raw_second):
    first = "first" if raw_first else [{"type": "text", "text": "first"}]
    second = "second" if raw_second else [{"type": "text", "text": "second"}]
    split = [{"role": role, "content": first}, {"role": role, "content": second}]
    combined = [
        {
            "role": role,
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    assert render(split) == render(combined)


def test_text_can_follow_all_parallel_results_in_next_user_message():
    split = history() + [
        *[{"role": "user", "content": [result]} for result in RESULTS],
        {"role": "user", "content": "Now sum the values."},
    ]
    combined = history() + [
        {
            "role": "user",
            "content": [*RESULTS, {"type": "text", "text": "Now sum the values."}],
        }
    ]
    assert render(split) == render(combined)


@pytest.mark.parametrize(
    "tail,match",
    [
        ([{"role": "user", "content": RESULTS[:1]}], "without `tool_result`"),
        (
            [
                {"role": "user", "content": [RESULTS[0]]},
                {"role": "user", "content": [RESULTS[0], RESULTS[1]]},
            ],
            "multiple `tool_result`",
        ),
        (
            [
                {"role": "user", "content": [RESULTS[0]]},
                {"role": "user", "content": "interrupted"},
                {"role": "user", "content": [RESULTS[1]]},
            ],
            "without `tool_result`",
        ),
        (
            [
                {"role": "user", "content": [RESULTS[0]]},
                {"role": "assistant", "content": "interrupted"},
                {"role": "user", "content": [RESULTS[1]]},
            ],
            "without `tool_result`",
        ),
        (
            [
                {"role": "user", "content": [RESULTS[0]]},
                {"role": "system", "content": "boundary"},
                {"role": "user", "content": [RESULTS[1]]},
            ],
            "without `tool_result`",
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "unknown",
                            "content": "x",
                        }
                    ],
                }
            ],
            "unexpected",
        ),
    ],
)
def test_invalid_tool_histories_still_fail(tail, match):
    with pytest.raises(ConversionError, match=match):
        convert(history() + tail)


def test_duplicate_tool_ids_across_assistant_messages_fail():
    with pytest.raises(ConversionError, match="must be unique"):
        convert(
            [
                {"role": "assistant", "content": [USES[0]]},
                {"role": "assistant", "content": [USES[0]]},
                {"role": "user", "content": [RESULTS[0]]},
            ]
        )


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_empty_block_message_is_not_hidden_by_coalescing(role):
    with pytest.raises(ConversionError, match="messages.1: all messages"):
        convert([{"role": role, "content": "valid"}, {"role": role, "content": []}])
