use deepseek_recipe::stream::state_machine::ParsingOptions;
use deepseek_recipe::stream::{
    ChunkGenerator, CompletionUsage, InferenceChunk, OutputChunk, StreamProcessor,
};
use tokio_stream::StreamExt;

struct IdentityGenerator;

impl ChunkGenerator for IdentityGenerator {
    type Chunk = OutputChunk;

    async fn generate(&mut self, chunk: OutputChunk) -> Vec<Self::Chunk> {
        vec![chunk]
    }
}

async fn json_output(parts: Vec<(String, usize)>) -> (String, CompletionUsage) {
    let input = tokio_stream::iter(parts.into_iter().map(|(content, content_tokens)| {
        InferenceChunk::Text {
            content,
            content_tokens,
        }
    }));
    let chunks: Vec<_> = StreamProcessor::new(
        IdentityGenerator,
        ParsingOptions {
            parse_json_output: true,
            ..ParsingOptions::default()
        },
    )
    .process(input)
    .map(|item| item.expect("text input cannot fail"))
    .collect()
    .await;

    let mut content = String::new();
    let mut usage = None;
    for chunk in chunks {
        match chunk {
            OutputChunk::Raw { content: delta } => content.push_str(&delta),
            OutputChunk::Finish {
                usage: final_usage, ..
            } => usage = Some(final_usage),
            _ => {}
        }
    }
    (content, usage.expect("the processor emits a finish chunk"))
}

#[tokio::test]
async fn json_output_is_independent_of_source_chunking() {
    let text = "prefix ```json\n{\"λ\":true}\n``` suffix";
    let expected = json_output(vec![(text.to_string(), 0)]).await;
    assert_eq!(expected.0, " {\"λ\":true} ");

    for split in text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
    {
        let actual = json_output(vec![
            (text[..split].to_string(), 0),
            (text[split..].to_string(), 0),
        ])
        .await;
        assert_eq!(actual, expected, "split={split}");
    }

    let characters = text.chars().map(|character| (character.to_string(), 0));
    let actual = json_output(characters.collect()).await;
    assert_eq!(actual, expected, "one source chunk per character");
}

#[tokio::test]
async fn empty_text_chunks_count_tokens_without_changing_json_output() {
    let text = "prefix ```json\n{\"ok\":true}\n``` suffix";
    let (content, usage) = json_output(vec![
        (String::new(), 2),
        (text.to_string(), 3),
        (String::new(), 0),
    ])
    .await;

    assert_eq!(content, " {\"ok\":true} ");
    assert_eq!(
        usage,
        CompletionUsage {
            completion_tokens: 5,
        }
    );
}
