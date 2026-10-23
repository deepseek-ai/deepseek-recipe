use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, InferenceTokenLogprob, OutputChunk,
    ParsingOptions, StreamProcessor, StreamTokenizer,
};
use tokio_stream::StreamExt;

struct AsciiTokenizer;
impl StreamTokenizer for AsciiTokenizer {
    fn decode_ids(&self, ids: &[u32], _: bool) -> Result<String, String> {
        Ok(ids.iter().map(|&id| char::from_u32(id).unwrap()).collect())
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        char::from_u32(id).map(|ch| ch.to_string())
    }
}
struct Passthrough;
impl ChunkGenerator for Passthrough {
    type Chunk = OutputChunk;
    async fn generate(&mut self, chunk: OutputChunk) -> Vec<OutputChunk> {
        vec![chunk]
    }
}
async fn run(text: &str, batch: bool, stop: Vec<String>) -> (String, Vec<String>) {
    let ids: Vec<u32> = text.chars().map(u32::from).collect();
    let batches = if batch {
        vec![ids]
    } else {
        ids.into_iter().map(|id| vec![id]).collect()
    };
    let mut input: Vec<InferenceChunk> = batches
        .into_iter()
        .map(|token_ids| {
            let logprobs = token_ids
                .iter()
                .map(|_| InferenceTokenLogprob {
                    logprob: -1.0,
                    top_logprobs: vec![],
                })
                .collect();
            InferenceChunk::Tokens {
                token_ids,
                logprobs: Some(logprobs),
            }
        })
        .collect();
    input.push(InferenceChunk::Finish {
        finish_reason: InferenceFinishReason::Stop,
    });
    let processor = StreamProcessor::new(
        Passthrough,
        ParsingOptions {
            stop_sequences: stop,
            ..Default::default()
        },
    )
    .with_tokenizer(AsciiTokenizer);
    let stream = processor.into_protocol_stream(tokio_stream::iter(input));
    let chunks: Vec<_> = std::pin::pin!(stream).collect().await;
    let mut text = String::new();
    let mut tokens = vec![];
    for chunk in chunks {
        if let OutputChunk::Content { content, logprobs } = chunk.unwrap() {
            text.push_str(&content);
            if let Some(entries) = logprobs {
                tokens.extend(entries.iter().map(|entry| entry.token.clone()));
            }
        }
    }
    (text, tokens)
}
#[tokio::test]
async fn stop_filtered_logprobs_do_not_depend_on_backend_batching() {
    let batched = run("ABSTOPZ", true, vec!["STOP".into()]).await;
    let single = run("ABSTOPZ", false, vec!["STOP".into()]).await;
    assert_eq!(batched.0, "AB");
    assert_eq!(batched, single);
    assert_eq!(batched.1, vec!["A", "B"]);
}
#[tokio::test]
async fn stripped_leading_newline_has_no_content_logprob() {
    let batched = run("\nAB", true, vec![]).await;
    let single = run("\nAB", false, vec![]).await;
    assert_eq!(batched.0, "AB");
    assert_eq!(batched, single);
}

#[tokio::test]
async fn json_padding_and_labels_do_not_duplicate_logprobs() {
    let text = "prefix{\"x\":1}\n```suffix";
    let ids: Vec<u32> = text.chars().map(u32::from).collect();
    for batch_size in 1..=ids.len() {
        let input = ids
            .chunks(batch_size)
            .map(|ids| InferenceChunk::Tokens {
                token_ids: ids.to_vec(),
                logprobs: Some(
                    ids.iter()
                        .map(|_| InferenceTokenLogprob {
                            logprob: -1.0,
                            top_logprobs: vec![],
                        })
                        .collect(),
                ),
            })
            .chain([InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop,
            }]);
        let processor = StreamProcessor::new(
            Passthrough,
            ParsingOptions {
                parse_json_output: true,
                ..Default::default()
            },
        )
        .with_tokenizer(AsciiTokenizer);
        let stream = processor.into_protocol_stream(tokio_stream::iter(input));
        let output: Vec<_> = std::pin::pin!(stream).collect().await;
        let mut tokens = Vec::new();
        for chunk in output {
            if let OutputChunk::Content {
                logprobs: Some(logprobs),
                ..
            } = chunk.unwrap()
            {
                tokens.extend(logprobs.iter().cloned());
            }
        }
        assert_eq!(
            tokens
                .iter()
                .map(|token| token.token.as_str())
                .collect::<String>(),
            "{\"x\":1}",
            "batch size {batch_size}"
        );
    }
}

struct WholePieceTokenizer(&'static str);
impl StreamTokenizer for WholePieceTokenizer {
    fn decode_ids(&self, _: &[u32], _: bool) -> Result<String, String> {
        Ok("ABSTOPZ".into())
    }
    fn id_to_token(&self, _: u32) -> Option<String> {
        Some(self.0.into())
    }
}
#[tokio::test]
async fn a_token_crossing_the_stop_boundary_is_not_fabricated_or_repeated() {
    let processor = StreamProcessor::new(
        Passthrough,
        ParsingOptions {
            stop_sequences: vec!["STOP".into()],
            ..Default::default()
        },
    )
    .with_tokenizer(WholePieceTokenizer("ABSTOPZ"));
    let input = tokio_stream::iter(vec![
        InferenceChunk::Tokens {
            token_ids: vec![1],
            logprobs: Some(vec![InferenceTokenLogprob {
                logprob: -0.25,
                top_logprobs: vec![],
            }]),
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ]);
    let output: Vec<_> = std::pin::pin!(processor.into_protocol_stream(input))
        .collect()
        .await;
    let content: Vec<_> = output
        .into_iter()
        .map(Result::unwrap)
        .filter_map(|chunk| {
            if let OutputChunk::Content { content, logprobs } = chunk {
                Some((content, logprobs))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0].0, "AB");
    assert!(content[0].1.is_none());
}

#[tokio::test]
async fn multibyte_token_logprobs_survive_stop_filtering_at_every_token_split() {
    let tokenizer = tokenizers::Tokenizer::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../static/tokenizers/v41/tokenizer.json"
    ))
    .unwrap();
    let text = "你好🤭STOPsuffix";
    let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    for split in 0..=ids.len() {
        let input = [&ids[..split], &ids[split..]]
            .into_iter()
            .map(|ids| InferenceChunk::Tokens {
                token_ids: ids.to_vec(),
                logprobs: Some(
                    ids.iter()
                        .map(|_| InferenceTokenLogprob {
                            logprob: -1.0,
                            top_logprobs: vec![],
                        })
                        .collect(),
                ),
            })
            .chain([InferenceChunk::Finish {
                finish_reason: InferenceFinishReason::Stop,
            }]);
        let processor = StreamProcessor::new(
            Passthrough,
            ParsingOptions {
                stop_sequences: vec!["STOP".into()],
                ..Default::default()
            },
        )
        .with_tokenizer(tokenizer.clone());
        let output: Vec<_> =
            std::pin::pin!(processor.into_protocol_stream(tokio_stream::iter(input)))
                .collect()
                .await;
        let mut content = String::new();
        let mut bytes = Vec::new();
        for chunk in output {
            if let OutputChunk::Content {
                content: part,
                logprobs,
            } = chunk.unwrap()
            {
                content.push_str(&part);
                if let Some(logprobs) = logprobs {
                    bytes.extend(
                        logprobs
                            .iter()
                            .flat_map(|token| token.bytes.iter().copied()),
                    );
                }
            }
        }
        assert_eq!(content, "你好🤭", "split {split}");
        assert_eq!(bytes, "你好🤭".as_bytes(), "split {split}");
    }
}

#[tokio::test]
async fn vocabulary_bytes_that_do_not_match_decoded_text_are_not_reported() {
    let processor = StreamProcessor::new(Passthrough, ParsingOptions::default())
        .with_tokenizer(WholePieceTokenizer("different bytes"));
    let input = tokio_stream::iter(vec![
        InferenceChunk::Tokens {
            token_ids: vec![1],
            logprobs: Some(vec![InferenceTokenLogprob {
                logprob: -0.25,
                top_logprobs: vec![],
            }]),
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ]);
    let output: Vec<_> = std::pin::pin!(processor.into_protocol_stream(input))
        .collect()
        .await;
    for chunk in output {
        if let OutputChunk::Content { content, logprobs } = chunk.unwrap() {
            assert_eq!(content, "ABSTOPZ");
            assert!(logprobs.is_none());
        }
    }
}

async fn public_chunks<G: ChunkGenerator>(
    generator: G,
    options: ParsingOptions,
) -> Vec<serde_json::Value>
where
    G::Chunk: serde::Serialize + Send,
{
    let ids: Vec<u32> = "ABSTOPZ".chars().map(u32::from).collect();
    let input = tokio_stream::iter(vec![
        InferenceChunk::Tokens {
            logprobs: Some(
                ids.iter()
                    .map(|_| InferenceTokenLogprob {
                        logprob: -1.0,
                        top_logprobs: vec![],
                    })
                    .collect(),
            ),
            token_ids: ids,
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ]);
    let processor = StreamProcessor::new(generator, options).with_tokenizer(AsciiTokenizer);
    std::pin::pin!(processor.into_protocol_stream(input))
        .map(|chunk| serde_json::to_value(chunk.unwrap()).unwrap())
        .collect()
        .await
}
#[tokio::test]
async fn chat_completion_wire_logprobs_match_retained_content() {
    use deepseek_recipe::openai::ChatCompletionRequest;
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model":"DeepSeek-V4.1-Flash", "messages":[{"role":"user","content":"Hello"}],
        "thinking":{"type":"disabled"}, "logprobs":true, "stop":"STOP"
    }))
    .unwrap();
    let converted = request.convert(ConversionOptions::default()).unwrap();
    let generator = ChatCompletionRequest::chunk_generator(
        &converted,
        "test-id".into(),
        "DeepSeek-V4.1-Flash".into(),
    );
    let chunks = public_chunks(generator, converted.parsing_options).await;
    let text: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    let tokens: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["logprobs"]["content"].as_array())
        .flatten()
        .map(|entry| entry["token"].as_str().unwrap())
        .collect();
    assert_eq!(text, "AB");
    assert_eq!(tokens, "AB");
}
#[tokio::test]
async fn responses_wire_logprobs_match_retained_content() {
    use deepseek_recipe::openai::ResponsesRequest;
    use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
    let request: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model":"DeepSeek-V4.1-Flash", "input":"Hello", "reasoning":{"effort":"none"}, "top_logprobs":0
    })).unwrap();
    let mut converted = request.convert(ConversionOptions::default()).unwrap();
    // Responses has no request-level stop field; the caller can configure parsing.
    converted.parsing_options.stop_sequences = vec!["STOP".into()];
    let generator = ResponsesRequest::chunk_generator(
        &converted,
        "test-id".into(),
        "DeepSeek-V4.1-Flash".into(),
    );
    let chunks = public_chunks(generator, converted.parsing_options).await;
    let deltas: Vec<_> = chunks
        .iter()
        .filter(|chunk| chunk["type"] == "response.output_text.delta")
        .collect();
    let text: String = deltas
        .iter()
        .map(|chunk| chunk["delta"].as_str().unwrap())
        .collect();
    let tokens: String = deltas
        .iter()
        .flat_map(|chunk| chunk["logprobs"].as_array().unwrap())
        .map(|entry| entry["token"].as_str().unwrap())
        .collect();
    assert_eq!(text, "AB");
    assert_eq!(tokens, "AB");
}
