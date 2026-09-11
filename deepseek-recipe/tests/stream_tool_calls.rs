use deepseek_recipe::openai::chat_completion::response::{
    ChatCompletionChunkGenerator, ChatCompletionFinishReason, ChatCompletionResponse,
};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::state_machine::ParsingOptions;
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, StreamProcessor};
use deepseek_recipe::util::append_delta::AppendDelta;
use serde_json::{Value, json};
use tokio_stream::{StreamExt, iter};

async fn assert_tool_arguments(chunks: Vec<String>, expected: &Value) {
    let generator = ChatCompletionChunkGenerator::new("id".into(), "model".into(), false, false);
    let processor = StreamProcessor::new(generator, ParsingOptions::default());
    let inference = chunks
        .into_iter()
        .map(|content| InferenceChunk::Text {
            content,
            content_tokens: 0,
        })
        .chain(std::iter::once(InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        }));
    let mut stream = std::pin::pin!(processor.process(iter(inference)));
    let mut response = ChatCompletionResponse::new("id".into(), "model".into(), 0, 0, 0);
    while let Some(chunk) = stream.next().await {
        response.append(chunk.unwrap());
    }
    let [choice] = response.choices.as_slice() else {
        panic!("expected one completion");
    };
    assert_eq!(
        choice.finish_reason,
        Some(ChatCompletionFinishReason::ToolCalls)
    );
    let [call] = choice.message.tool_calls.as_deref().unwrap() else {
        panic!("expected one tool call");
    };
    assert_eq!(call.function.name, "submit");
    assert_eq!(
        serde_json::from_str::<Value>(&call.function.arguments).unwrap(),
        *expected
    );
}

#[tokio::test]
async fn quoted_tool_parameter_names_survive_chunk_boundaries() {
    let expected = json!({
        "shipping address": "Paris\n日本",
        " string= mode ": {"enabled": true},
        "count": 2,
        "": "empty key",
    });
    // V4 and V4.1 use the same quoted attributes, with different DSML tag names.
    for (block, prefix) in [("tool_calls", ""), (" calls", " ")] {
        let output = format!(
            "<｜DSML｜{block}>\n\
             <｜DSML｜{prefix}invoke name=\"submit\">\n\
             <｜DSML｜{prefix}parameter name=\"shipping address\" string=\"true\">Paris\n日本</｜DSML｜{prefix}parameter>\n\
             <｜DSML｜{prefix}parameter name=\" string= mode \" string=\"false\">{{\"enabled\":true}}</｜DSML｜{prefix}parameter>\n\
             <｜DSML｜{prefix}parameter name=\"count\" string=\"false\">2</｜DSML｜{prefix}parameter>\n\
             <｜DSML｜{prefix}parameter name=\"\" string=\"true\">empty key</｜DSML｜{prefix}parameter>\n\
             </｜DSML｜{prefix}invoke>\n\
             </｜DSML｜{block}>"
        );
        assert_tool_arguments(vec![output.clone()], &expected).await;
        for (split, _) in output.char_indices().skip(1) {
            assert_tool_arguments(
                vec![output[..split].to_string(), output[split..].to_string()],
                &expected,
            )
            .await;
        }
        assert_tool_arguments(output.chars().map(|ch| ch.to_string()).collect(), &expected).await;
    }
}
