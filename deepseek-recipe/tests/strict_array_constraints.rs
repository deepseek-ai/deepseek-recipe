use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{
    ConversationRequest, ConversionOptions, OutputConstraint, ProtocolRequest,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

fn convert<T: ProtocolRequest + DeserializeOwned>(request: Value) -> ConversationRequest {
    serde_json::from_value::<T>(request)
        .expect("the request deserializes")
        .convert(ConversionOptions::default())
        .expect("the request converts")
}

#[test]
fn strict_array_constraints_survive_protocol_conversion() {
    let schema = json!({
        "type": "object",
        "properties": {
            "labels": {
                "type": "array", "items": {"type": "string"},
                "minItems": 2, "maxItems": 3, "uniqueItems": true,
            },
        },
    });
    let model = "DeepSeek-V4.1-Flash";
    let messages = json!([{"role": "user", "content": "Return JSON labels."}]);
    let function = json!({"name": "label", "parameters": schema, "strict": true});
    let requests = [
        convert::<ChatCompletionRequest>(json!({
            "model": model, "messages": messages,
            "tools": [{"type": "function", "function": function}],
        })),
        convert::<ResponsesRequest>(json!({
            "model": model, "input": "Return labels.",
            "tools": [{"type": "function", "name": "label", "parameters": schema, "strict": true}],
        })),
        convert::<ChatCompletionRequest>(json!({
            "model": model, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "labels", "schema": schema, "strict": true,
            }},
        })),
        convert::<ResponsesRequest>(json!({
            "model": model, "input": "Return JSON labels.",
            "text": {"format": {"type": "json_schema", "name": "labels", "schema": schema, "strict": true}},
        })),
    ];

    for request in requests {
        let normalized = match request.inference_options.output_constraint.unwrap() {
            OutputConstraint::ToolCalls { tools, .. } => {
                assert_eq!(
                    request.conversation.tools[0].parameters,
                    tools[0].parameters
                );
                tools[0].parameters.clone()
            }
            OutputConstraint::Json {
                schema: Some(schema),
            } => schema,
            other => panic!("unexpected constraint: {other:?}"),
        };
        let validator = jsonschema::validator_for(&normalized).expect("valid normalized schema");
        for labels in [json!(["a", "b"]), json!(["a", "b", "c"])] {
            assert!(validator.is_valid(&json!({"labels": labels})));
        }
        for labels in [
            json!([]),
            json!(["a"]),
            json!(["a", "b", "c", "d"]),
            json!(["a", "a"]),
        ] {
            assert!(
                !validator.is_valid(&json!({"labels": labels})),
                "normalized schema accepted invalid labels {labels}: {normalized}"
            );
        }
    }
}
