use std::collections::HashMap;

use deepseek_recipe::openai::ResponsesRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_core::messages::InputMessage;
use serde_json::{Value, json};

fn convert_history(
    model: &str,
    bare_name: &str,
    reversed: bool,
    declared: bool,
) -> (HashMap<String, String>, Vec<String>) {
    let mut calls = vec![
        json!({"type": "function_call", "call_id": "namespaced", "namespace": "ns", "name": "run", "arguments": "{}"}),
        json!({"type": "function_call", "call_id": "bare", "name": bare_name, "arguments": "{}"}),
    ];
    if reversed {
        calls.reverse();
    }
    calls.extend([
        json!({"type": "function_call_output", "call_id": "namespaced", "output": "namespace result"}),
        json!({"type": "function_call_output", "call_id": "bare", "output": "top-level result"}),
        json!({"role": "user", "content": "continue"}),
    ]);
    let tools: Vec<Value> = if declared {
        vec![
            json!({"type": "namespace", "name": "ns", "tools": [{"type": "function", "name": "run"}]}),
        ]
    } else {
        Vec::new()
    };
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": model,
        "reasoning": {"effort": "none"},
        "input": calls,
        "tools": tools,
    }))
    .unwrap();
    let converted = request.convert(ConversionOptions::default()).unwrap();
    let mut names = HashMap::new();
    for message in &converted.conversation.messages {
        if let InputMessage::Assistant {
            tool_calls: Some(calls),
            ..
        } = message
        {
            for call in calls {
                names.insert(call.id.clone(), call.name.clone());
            }
        }
    }
    if declared {
        let mapped = converted
            .response_options
            .namespace_tool_map
            .as_ref()
            .unwrap();
        assert_eq!(
            mapped.get(&names["namespaced"]),
            Some(&("ns".to_owned(), "run".to_owned()))
        );
    }
    let declarations = converted
        .conversation
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    (names, declarations)
}

#[test]
fn a_historical_custom_call_reserves_its_original_name() {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "DeepSeek-V4-Pro",
        "reasoning": {"effort": "none"},
        "tools": [{"type": "namespace", "name": "apply_", "tools": [{"type": "function", "name": "patch"}]}],
        "input": [
            {"type": "custom_tool_call", "call_id": "old_patch", "name": "apply_patch", "input": "old patch"},
            {"type": "custom_tool_call_output", "call_id": "old_patch", "output": "done"},
            {"role": "user", "content": "continue"}
        ]
    })).unwrap();
    let converted = request.convert(ConversionOptions::default()).unwrap();
    assert_ne!(converted.conversation.tools[0].name, "apply_patch");
    let historical = converted
        .conversation
        .messages
        .iter()
        .find_map(|message| {
            if let InputMessage::Assistant {
                tool_calls: Some(calls),
                ..
            } = message
            {
                calls.first()
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(historical.name, "apply_patch");
}

#[test]
fn undeclared_historical_namespace_does_not_alias_a_bare_tool() {
    for (model, bare) in [
        ("DeepSeek-V4-Pro", "ns__run"),
        ("DeepSeek-V4.1-Flash", "ns::run"),
    ] {
        for reversed in [false, true] {
            let (names, _) = convert_history(model, bare, reversed, false);
            assert_eq!(names["bare"], bare);
            assert_ne!(
                names["namespaced"], names["bare"],
                "{model}, reversed={reversed}"
            );
        }
    }
}

#[test]
fn current_namespace_reserves_historical_bare_names_before_flattening() {
    for (model, bare) in [
        ("DeepSeek-V4-Pro", "ns__run"),
        ("DeepSeek-V4.1-Flash", "ns::run"),
    ] {
        for reversed in [false, true] {
            let (names, declarations) = convert_history(model, bare, reversed, true);
            assert_eq!(names["bare"], bare);
            assert_ne!(
                names["namespaced"], names["bare"],
                "{model}, reversed={reversed}"
            );
            assert_eq!(declarations, vec![names["namespaced"].clone()]);
        }
    }
}
