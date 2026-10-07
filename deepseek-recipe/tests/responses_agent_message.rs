use deepseek_recipe::openai::ResponsesRequest;
use deepseek_recipe::request::{ConversionError, ConversionOptions, ProtocolRequest};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::multimodal::IMAGE_SPECIAL_TOKEN;
use serde_json::{Value, json};

fn agent_message(author: &str, recipient: &str, content: Value) -> Value {
    json!({
        "type": "agent_message",
        "id": "amsg_test",
        "author": author,
        "recipient": recipient,
        "content": content,
        "internal_chat_message_metadata_passthrough": {"turn_id": "test-turn"},
    })
}

fn convert(input: Vec<Value>) -> Result<Vec<InputMessage>, ConversionError> {
    let request: ResponsesRequest = serde_json::from_value(json!({
        "model": "test-model",
        "input": input,
    }))
    .unwrap();
    request
        .convert(ConversionOptions::default())
        .map(|converted| converted.conversation.messages)
}

#[test]
fn plaintext_agent_tasks_messages_and_results_remain_in_history_order() {
    let mut input = vec![json!({"role": "user", "content": "Before agents"})];
    for (kind, author, recipient) in [
        ("NEW_TASK", "/root", "/root/worker"),
        ("MESSAGE", "/root", "/root/worker"),
        ("FINAL_ANSWER", "/root/worker", "/root"),
    ] {
        input.push(agent_message(
            author,
            recipient,
            json!([
                {"type": "input_text", "text": format!("Message Type: {kind}\nPayload:")},
                {"type": "input_text", "text": format!("{kind} marker: 世界 🦀")},
            ]),
        ));
    }
    input.push(json!({"role": "user", "content": "After agents"}));
    let messages = convert(input).unwrap();
    assert_eq!(messages.len(), 5);
    for (index, (kind, author, recipient)) in [
        ("NEW_TASK", "/root", "/root/worker"),
        ("MESSAGE", "/root", "/root/worker"),
        ("FINAL_ANSWER", "/root/worker", "/root"),
    ]
    .into_iter()
    .enumerate()
    {
        let InputMessage::User {
            content,
            image_sources,
        } = &messages[index + 1]
        else {
            panic!("An incoming agent message must be user input");
        };
        assert!(content.contains(author));
        assert!(content.contains(recipient));
        assert!(content.contains(&format!(
            "Message Type: {kind}\nPayload:\n{kind} marker: 世界 🦀"
        )));
        assert!(image_sources.is_empty());
    }
    assert!(
        matches!(&messages[0], InputMessage::User { content, .. } if content == "Before agents")
    );
    assert!(
        matches!(&messages[4], InputMessage::User { content, .. } if content == "After agents")
    );
}

#[test]
fn agent_messages_preserve_completed_tool_groups() {
    let messages = convert(vec![
        json!({"type": "function_call", "call_id": "check", "name": "check", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "check", "output": "CHECK_COMPLETED"}),
        agent_message("/root/worker", "/root", json!([{"type": "input_text", "text": "CHILD_RESULT"}])),
        json!({"type": "custom_tool_call", "call_id": "patch", "name": "apply_patch", "input": "patch text"}),
        json!({"type": "custom_tool_call_output", "call_id": "patch", "output": "PATCH_COMPLETED"}),
        agent_message("/root", "/root/worker", json!([{"type": "input_text", "text": "FOLLOWUP_TASK"}])),
    ]).unwrap();
    assert_eq!(messages.len(), 6);
    assert!(
        matches!(&messages[0], InputMessage::Assistant { tool_calls: Some(calls), .. } if calls[0].id == "check")
    );
    assert!(
        matches!(&messages[1], InputMessage::Tool { content, .. } if content == "CHECK_COMPLETED")
    );
    assert!(
        matches!(&messages[2], InputMessage::User { content, .. } if content.contains("CHILD_RESULT"))
    );
    assert!(
        matches!(&messages[3], InputMessage::Assistant { tool_calls: Some(calls), .. } if calls[0].id == "patch")
    );
    assert!(
        matches!(&messages[4], InputMessage::Tool { content, .. } if content == "PATCH_COMPLETED")
    );
    assert!(
        matches!(&messages[5], InputMessage::User { content, .. } if content.contains("FOLLOWUP_TASK"))
    );
}

#[test]
fn agent_messages_do_not_bypass_unresolved_tool_calls() {
    let error = convert(vec![
        json!({"type": "function_call", "call_id": "pending", "name": "check", "arguments": "{}"}),
        agent_message(
            "/root/worker",
            "/root",
            json!([{"type": "input_text", "text": "CHILD_RESULT"}]),
        ),
        json!({"type": "function_call_output", "call_id": "pending", "output": "CHECK_COMPLETED"}),
    ])
    .unwrap_err();
    assert!(matches!(error, ConversionError::BadRequest { .. }));
}

#[test]
fn encrypted_mixed_unknown_and_empty_agent_content_is_rejected() {
    for content in [
        json!([]),
        json!([{"type": "input_text", "text": "  \n"}]),
        json!([{"type": "encrypted_content", "encrypted_content": "secret-ciphertext"}]),
        json!([
            {"type": "input_text", "text": "VISIBLE_PREFIX"},
            {"type": "encrypted_content", "encrypted_content": "secret-ciphertext"},
        ]),
        json!([{"type": "unknown_content", "text": "OPAQUE"}]),
        json!([{"type": "output_text", "text": "Not agent input text"}]),
        json!([{"type": "input_image", "image_url": "https://example.com/image.png"}]),
        json!([{"type": "input_file", "file_id": "file-test"}]),
    ] {
        let error = convert(vec![agent_message("/root", "/root/worker", content)]).unwrap_err();
        assert!(matches!(error, ConversionError::BadRequest { .. }));
        assert!(!error.to_string().contains("secret-ciphertext"));
        assert!(!error.to_string().contains("VISIBLE_PREFIX"));
    }
    for (author, recipient) in [("", "/root"), ("/root", " \n")] {
        assert!(
            convert(vec![agent_message(
                author,
                recipient,
                json!([{"type": "input_text", "text": "TASK"}])
            )])
            .is_err()
        );
    }
}

#[test]
fn malformed_agent_items_fail_deserialization() {
    for content in [
        Value::Null,
        json!("TASK"),
        json!([{"type": "input_text", "text": 42}]),
    ] {
        let request = json!({"model": "test-model", "input": [agent_message("/root", "/root/worker", content)]});
        assert!(serde_json::from_value::<ResponsesRequest>(request).is_err());
    }
    for field in ["author", "recipient", "content"] {
        let mut item = agent_message(
            "/root",
            "/root/worker",
            json!([{"type": "input_text", "text": "TASK"}]),
        );
        item.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<ResponsesRequest>(
                json!({"model": "test-model", "input": [item]})
            )
            .is_err()
        );
    }
}

#[test]
fn agent_text_and_attribution_follow_image_placeholder_validation() {
    for (author, recipient, text) in [
        ("/root", "/root/worker", IMAGE_SPECIAL_TOKEN),
        (IMAGE_SPECIAL_TOKEN, "/root/worker", "TASK"),
        ("/root", IMAGE_SPECIAL_TOKEN, "TASK"),
    ] {
        let error = convert(vec![agent_message(
            author,
            recipient,
            json!([{"type": "input_text", "text": text}]),
        )])
        .unwrap_err();
        assert!(matches!(error, ConversionError::BadRequest { .. }));
    }
}

#[test]
fn unrelated_unsupported_items_keep_their_existing_behavior() {
    let messages = convert(vec![
        json!({"type": "web_search_call"}),
        json!({"role": "user", "content": "Continue"}),
    ])
    .unwrap();
    assert_eq!(
        messages,
        vec![InputMessage::User {
            content: "Continue".into(),
            image_sources: vec![]
        }]
    );
}
