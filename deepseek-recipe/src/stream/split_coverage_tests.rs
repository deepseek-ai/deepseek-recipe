//! Chunk-split and end-of-stream coverage for stream output parsing.
//!
//! `OutputParser::feed` retains partial marker matches across chunks, so
//! marker recognition, DSML tool-call parsing, reasoning/content staging, and
//! stop-sequence matching are chunk-transparent by design. Other behavior is
//! chunk-dependent by contract: text replaced around JSON output becomes one
//! space for each source chunk it spans, and completion usage counts the
//! whole chunk containing a matched stop sequence. These tests pin both
//! sides of that contract: every scenario runs as a single chunk, at every
//! two-way split on a character boundary, and at a seeded sample of k-way
//! splits, and the parsed output is compared accordingly.

use deepseek_recipe_core::special_tokens;
use tokenizers::Tokenizer;
use tokio_stream::{StreamExt, iter};

use super::inference::{InferenceChunk, InferenceFinishReason};
use super::output_parser::ThinkingInitialStage;
use super::{
    ChunkGenerator, CompletionUsage, FinishReason, OutputChunk, ParsingOptions, StreamProcessor,
    StreamTokenizer,
};

/// A chunk generator whose protocol chunks are the parsed output chunks,
/// mirroring the processor's own test generator.
struct Recorder;

impl ChunkGenerator for Recorder {
    type Chunk = OutputChunk;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<OutputChunk> {
        vec![chunk]
    }
}

/// Merge adjacent same-kind text runs. The processor coalesces output for
/// the protocol adapters at its own granularity, which may legitimately vary
/// with the split; the concatenated text may not.
fn normalize(chunks: Vec<OutputChunk>) -> Vec<OutputChunk> {
    let mut merged: Vec<OutputChunk> = Vec::new();
    for chunk in chunks {
        match (merged.last_mut(), &chunk) {
            (
                Some(OutputChunk::Content {
                    content,
                    logprobs: None,
                }),
                OutputChunk::Content {
                    content: next,
                    logprobs: None,
                },
            ) => content.push_str(next),
            (
                Some(OutputChunk::Reasoning {
                    content,
                    logprobs: None,
                }),
                OutputChunk::Reasoning {
                    content: next,
                    logprobs: None,
                },
            ) => content.push_str(next),
            (
                Some(OutputChunk::ToolCallArgumentsDelta { arguments }),
                OutputChunk::ToolCallArgumentsDelta { arguments: next },
            ) => arguments.push_str(next),
            _ => merged.push(chunk),
        }
    }
    merged
}

/// Replace the completion usage of a finish chunk with the default, for
/// scenarios whose usage is chunk-dependent by contract (stop sequences).
fn without_usage(chunks: Vec<OutputChunk>) -> Vec<OutputChunk> {
    chunks
        .into_iter()
        .map(|chunk| match chunk {
            OutputChunk::Finish {
                finish_reason,
                stop_sequence,
                ..
            } => OutputChunk::Finish {
                finish_reason,
                stop_sequence,
                completion_usage: CompletionUsage::default(),
            },
            chunk => chunk,
        })
        .collect()
}

fn text_chunks(pieces: &[&str]) -> Vec<InferenceChunk> {
    pieces
        .iter()
        .map(|piece| InferenceChunk::Text {
            content: (*piece).to_string(),
            token_count: piece.len(),
        })
        .chain([InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        }])
        .collect()
}

async fn run(options: &ParsingOptions, chunks: Vec<InferenceChunk>) -> Vec<OutputChunk> {
    let stream = StreamProcessor::new(Recorder, options.clone()).into_protocol_stream(iter(chunks));
    let mut stream = std::pin::pin!(stream);
    let mut output = Vec::new();
    while let Some(item) = stream.next().await {
        output.push(item.expect("these scenarios never fail stream processing"));
    }
    output
}

fn char_boundaries(text: &str) -> Vec<usize> {
    text.char_indices()
        .map(|(index, _)| index)
        .filter(|index| *index > 0 && *index < text.len())
        .collect()
}

/// Split `text` at the given byte offsets into pieces.
fn cut(text: &str, mut points: Vec<usize>) -> Vec<&str> {
    points.sort_unstable();
    points.dedup();
    let mut pieces = Vec::with_capacity(points.len() + 1);
    let mut start = 0;
    for point in points {
        pieces.push(&text[start..point]);
        start = point;
    }
    pieces.push(&text[start..]);
    pieces
}

/// A tiny deterministic RNG for sampling k-way splits, avoiding a
/// dev-dependency for one test helper.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }
}

/// Sample k-way splits (k in 2..=5) of `text` at character boundaries.
fn sampled_splits(text: &str, seed: u64, samples: usize) -> Vec<Vec<usize>> {
    let boundaries = char_boundaries(text);
    let mut rng = XorShift(seed.max(1));
    (0..samples)
        .map(|_| {
            let cuts = 2 + (rng.next() % 4) as usize;
            (0..cuts)
                .map(|_| boundaries[(rng.next() as usize) % boundaries.len()])
                .collect()
        })
        .collect()
}

/// How the parsed output of a scenario is compared across splits.
#[derive(Clone)]
enum Oracle {
    /// The normalized output is identical for every split, and the completion
    /// usage equals the transcript's byte count (the per-piece `token_count`).
    Invariant,
    /// JSON mode: text before the JSON is documented to become one space per
    /// source chunk it spans. The expected content is computed per split.
    /// If the chunk-independent spacing of #14 is ever adopted, this oracle
    /// flips to one space per replaced run.
    JsonSpacing {
        prefix_len: usize,
        json: &'static str,
    },
    /// A stop sequence ends output: emitted text is identical across splits,
    /// while completion usage counts the whole stop-containing chunk by
    /// contract and is therefore excluded from the comparison.
    StopSequence,
}

struct Scenario {
    name: &'static str,
    options: ParsingOptions,
    transcript: &'static str,
    oracle: Oracle,
}

fn reasoning_options() -> ParsingOptions {
    ParsingOptions {
        thinking_initial_stage: Some(ThinkingInitialStage::ReasoningStart),
        ..ParsingOptions::default()
    }
}

fn json_options() -> ParsingOptions {
    ParsingOptions {
        parse_tool_calls: false,
        parse_json_output: true,
        ..ParsingOptions::default()
    }
}

const JSON_PREFIX: &str = "Here you go: ";
const FENCED_JSON: &str = "{\"a\": 1, \"b\": \"🤭\"}";
const BARE_JSON: &str = "{\"a\": 1}";

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "plain content",
            options: ParsingOptions::default(),
            transcript: "The answer is 42.\nSecond line 🤭 北京。",
            oracle: Oracle::Invariant,
        },
        Scenario {
            name: "reasoning then answer",
            options: reasoning_options(),
            transcript: "Let me think step by step.\n🤭 北京</think>The answer is 42.",
            oracle: Oracle::Invariant,
        },
        Scenario {
            name: "tool calls",
            options: ParsingOptions::default(),
            transcript: "Let's call the tool 🤭:\n<｜DSML｜tool_calls>\n\
                         <｜DSML｜invoke name=\"get_current_weather\">\n\
                         </｜DSML｜invoke>\n\
                         <｜DSML｜invoke name=\"get_current_weather\">\n\
                         <｜DSML｜parameter name=\"location\" string=\"true\">{\"value\": \"北京\"}</｜DSML｜parameter>\n\
                         <｜DSML｜parameter name=\"unit\" string=\"false\">{\"value\": \"celsius\"}</｜DSML｜parameter>\n\
                         </｜DSML｜invoke>\n\
                         </｜DSML｜tool_calls>",
            oracle: Oracle::Invariant,
        },
        Scenario {
            name: "fenced json",
            options: json_options(),
            transcript: "Here you go: ```json\n{\"a\": 1, \"b\": \"🤭\"}\n```",
            oracle: Oracle::JsonSpacing {
                prefix_len: JSON_PREFIX.len(),
                json: FENCED_JSON,
            },
        },
        Scenario {
            name: "bare json",
            options: json_options(),
            transcript: "Here you go: {\"a\": 1}",
            oracle: Oracle::JsonSpacing {
                prefix_len: JSON_PREFIX.len(),
                json: BARE_JSON,
            },
        },
        Scenario {
            name: "stop sequence",
            options: ParsingOptions {
                parse_tool_calls: false,
                stop_sequences: vec!["STOP".to_string()],
                ..ParsingOptions::default()
            },
            transcript: "The answer is 42.STOP trailing text is never read",
            oracle: Oracle::StopSequence,
        },
    ]
}

/// The completion usage a text-mode scenario must report: the byte count of
/// the transcript, conserved across splits by the per-piece `token_count`.
fn expected_usage(transcript: &str) -> CompletionUsage {
    CompletionUsage {
        completion_tokens: transcript.len(),
        reasoning_tokens: None,
    }
}

fn usage_of(chunks: &[OutputChunk]) -> CompletionUsage {
    chunks
        .iter()
        .find_map(|chunk| match chunk {
            OutputChunk::Finish {
                completion_usage, ..
            } => Some(*completion_usage),
            _ => None,
        })
        .expect("every run ends with a finish chunk")
}

/// The content a JSON-mode scenario documents for one split: one space for
/// each piece that spans bytes before the JSON, then the JSON text.
fn expected_json_content(prefix_len: usize, json: &str, pieces: &[&str]) -> String {
    let mut spaces = 0;
    let mut start = 0;
    for piece in pieces {
        if start < prefix_len && !piece.is_empty() {
            spaces += 1;
        }
        start += piece.len();
    }
    " ".repeat(spaces) + json
}

async fn check_split(scenario: &Scenario, pieces: Vec<&str>, context: String) {
    let chunks = text_chunks(&pieces);
    let output = run(&scenario.options, chunks).await;
    match &scenario.oracle {
        Oracle::Invariant => {
            let usage = usage_of(&output);
            let whole = run(&scenario.options, text_chunks(&[scenario.transcript])).await;
            assert_eq!(
                normalize(whole),
                normalize(output),
                "scenario {:?} diverged at {context}",
                scenario.name,
            );
            assert_eq!(
                usage,
                expected_usage(scenario.transcript),
                "scenario {:?} lost usage at {context}",
                scenario.name,
            );
        }
        Oracle::JsonSpacing { prefix_len, json } => {
            let content: String = normalize(output)
                .into_iter()
                .filter_map(|chunk| match chunk {
                    OutputChunk::Content { content, .. } => Some(content),
                    _ => None,
                })
                .collect();
            assert_eq!(
                content,
                expected_json_content(*prefix_len, json, &pieces),
                "scenario {:?} diverged at {context}",
                scenario.name,
            );
        }
        Oracle::StopSequence => {
            let whole = run(&scenario.options, text_chunks(&[scenario.transcript])).await;
            assert_eq!(
                without_usage(normalize(whole)),
                without_usage(normalize(output)),
                "scenario {:?} diverged at {context}",
                scenario.name,
            );
        }
    }
}

#[tokio::test]
async fn text_output_is_independent_of_two_way_splits() {
    for scenario in scenarios() {
        for cut_point in char_boundaries(scenario.transcript) {
            let pieces = cut(scenario.transcript, vec![cut_point]);
            check_split(&scenario, pieces, format!("split at byte {cut_point}")).await;
        }
    }
}

#[tokio::test]
async fn text_output_is_independent_of_sampled_k_way_splits() {
    for (index, scenario) in scenarios().into_iter().enumerate() {
        for points in sampled_splits(scenario.transcript, 0x5EED + index as u64, 100) {
            let pieces = cut(scenario.transcript, points.clone());
            check_split(&scenario, pieces, format!("split at {points:?}")).await;
        }
    }
}

#[tokio::test]
async fn text_output_is_independent_with_empty_chunks_interleaved() {
    for scenario in scenarios() {
        // Empty chunks around JSON output interact with the documented
        // per-chunk spacing; that combination is pinned by the JSON scenarios
        // above and by the resolution of #14, so it is excluded here.
        if matches!(&scenario.oracle, Oracle::JsonSpacing { .. }) {
            continue;
        }
        for cut_point in char_boundaries(scenario.transcript) {
            let pieces = cut(scenario.transcript, vec![cut_point]);
            let interleaved = [vec![""], pieces, vec![""]].concat();
            check_split(&scenario, interleaved, format!("split at byte {cut_point}")).await;
        }
    }
}

async fn run_tokenized<T: StreamTokenizer + Clone + 'static>(
    options: &ParsingOptions,
    tokenizer: T,
    chunks: Vec<InferenceChunk>,
) -> Vec<OutputChunk> {
    let stream = StreamProcessor::new(Recorder, options.clone())
        .with_tokenizer(tokenizer)
        .into_protocol_stream(iter(chunks));
    let mut stream = std::pin::pin!(stream);
    let mut output = Vec::new();
    while let Some(item) = stream.next().await {
        output.push(item.expect("these scenarios never fail decoding"));
    }
    output
}

fn token_chunks(ids: &[u32], cut_point: usize) -> Vec<InferenceChunk> {
    let mut chunks = vec![InferenceChunk::Tokens {
        token_ids: ids[..cut_point].to_vec(),
        logprobs: None,
    }];
    if cut_point < ids.len() {
        chunks.push(InferenceChunk::Tokens {
            token_ids: ids[cut_point..].to_vec(),
            logprobs: None,
        });
    }
    chunks.push(InferenceChunk::Finish {
        finish_reason: InferenceFinishReason::Stop,
    });
    chunks
}

#[tokio::test]
async fn token_output_is_independent_of_token_chunk_splits() {
    let tokenizer = v41_tokenizer();
    let transcript = "Hello world 🤭 北京。The answer is 42.";
    let ids = tokenizer
        .encode(transcript, false)
        .expect("the text encodes")
        .get_ids()
        .to_vec();
    let options = ParsingOptions::default();
    let whole =
        normalize(run_tokenized(&options, tokenizer.clone(), token_chunks(&ids, ids.len())).await);
    for cut_point in 1..ids.len() {
        let output =
            run_tokenized(&options, tokenizer.clone(), token_chunks(&ids, cut_point)).await;
        assert_eq!(
            whole,
            normalize(output),
            "token output diverged at token split {cut_point}",
        );
    }
}

fn v41_tokenizer() -> Tokenizer {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../static/tokenizers/v41/tokenizer.json"
    );
    Tokenizer::from_file(path).expect("the V4.1 tokenizer loads")
}

/// A two-token tokenizer whose second token decodes to the replacement
/// character, for the end-of-stream buffering case of #16.
#[derive(Clone)]
struct MojibakeTokenizer;

impl StreamTokenizer for MojibakeTokenizer {
    fn decode_ids(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String, String> {
        ids.iter()
            .map(|id| match id {
                1 => Ok('a'.to_string()),
                2 => Ok('\u{FFFD}'.to_string()),
                id => Err(format!("unknown token id {id}")),
            })
            .collect()
    }
}

/// Text that legitimately ends with U+FFFD stays buffered in the decoder,
/// waiting for a completion that never comes, and is dropped at the end of
/// the stream: neither text nor its token id is accounted. This pins the
/// behavior documented on `StreamDecoder`; when the end-of-stream flush of
/// #16 lands, flip the expectations to content "a\u{FFFD}" and two
/// completion tokens.
#[tokio::test]
async fn replacement_character_at_end_of_stream_is_dropped() {
    let chunks = vec![
        InferenceChunk::Tokens {
            token_ids: vec![1],
            logprobs: None,
        },
        InferenceChunk::Tokens {
            token_ids: vec![2],
            logprobs: None,
        },
        InferenceChunk::Finish {
            finish_reason: InferenceFinishReason::Stop,
        },
    ];
    let output = run_tokenized(&ParsingOptions::default(), MojibakeTokenizer, chunks).await;
    assert_eq!(
        normalize(output),
        vec![
            OutputChunk::Start {
                system_fingerprint: None,
                prompt_usage: Default::default(),
            },
            OutputChunk::Content {
                content: "a".to_string(),
                logprobs: None,
            },
            OutputChunk::Finish {
                finish_reason: FinishReason::Stop,
                stop_sequence: None,
                completion_usage: CompletionUsage {
                    completion_tokens: 1,
                    reasoning_tokens: None,
                },
            },
        ],
    );
}

/// The default special-token mapping is the V4 spelling this corpus relies
/// on; guard the transcripts against a silent default change.
#[test]
fn default_special_tokens_match_the_corpus() {
    let tokens = special_tokens::default_special_tokens();
    assert_eq!(
        tokens.token(special_tokens::SpecialToken::ThinkingEnd),
        "</think>"
    );
    assert_eq!(tokens.token(special_tokens::SpecialToken::Dsml), "｜DSML｜");
}
