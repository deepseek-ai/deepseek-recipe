use std::collections::VecDeque;
use std::pin::pin;

use async_stream::stream;
use tokio_stream::{Stream, StreamExt};

use super::decoder::{StreamDecoder, TokenizerDecoder};
use super::inference::{
    CompletionUsage, FinishReason, InferenceChunk, InferenceFinishReason, PromptUsage,
};
use super::state_machine::{OutputAction, OutputActionSegment, ParsingOptions, StateMachine};
use super::{ChunkGenerator, OutputChunk, StreamError};

/// Parse backend inference chunks and produce protocol response events.
pub struct StreamProcessor<G> {
    generator: G,
    options: ParsingOptions,
    decoder: Option<StreamDecoder>,
}

impl<G> StreamProcessor<G>
where
    G: ChunkGenerator,
{
    /// Combine a protocol event generator with output parsing options.
    pub fn new(generator: G, options: ParsingOptions) -> Self {
        Self {
            generator,
            options,
            decoder: None,
        }
    }

    /// Decode `InferenceChunk::Token` ids with the supplied decoder.
    ///
    /// Ids that contribute text count as completion tokens, including the ids
    /// buffered while a multi-token character was incomplete. IDs still buffered
    /// at the end of input contribute neither text nor completion usage.
    /// Without a decoder, a token chunk fails the stream with
    /// [`StreamError::MissingTokenizer`].
    pub fn with_tokenizer(mut self, decoder: impl TokenizerDecoder + 'static) -> Self {
        self.decoder = Some(StreamDecoder::new(Box::new(decoder)));
        self
    }

    /// Consume inference chunks until a finish chunk, matched stop sequence, or EOF.
    ///
    /// A ready chunk received before output starts supplies the initial prompt
    /// usage and fingerprint. Without it, the start event uses zero prompt usage
    /// and no fingerprint. Later ready chunks do not update emitted metadata.
    /// Completion usage accumulates each processed chunk's token count, including
    /// the entire chunk containing a stop sequence; later chunks are not read.
    /// Missing tokenizers and decoder failures yield an error and end the stream
    /// without a normal finish event.
    pub fn process(
        self,
        inference: impl Stream<Item = InferenceChunk> + Send,
    ) -> impl Stream<Item = Result<G::Chunk, StreamError>> + Send
    where
        G::Chunk: Send,
    {
        let mut generator = self.generator;
        let options = self.options;
        let mut decoder = self.decoder;
        stream! {
            let mut inference = pin!(inference);
            let mut state_machine = StateMachine::new(options);
            let mut stashed = StashedChunks::new();

            let mut started = false;
            let mut prompt_usage = PromptUsage::default();
            let mut completion_usage = CompletionUsage::default();
            let mut backend_finish = None;
            let mut stop_sequence = None;

            while let Some(chunk) = inference.next().await {
                let (content, content_tokens) = match chunk {
                    InferenceChunk::Ready {
                        system_fingerprint,
                        prompt_usage: ready_usage,
                    } => {
                        prompt_usage = ready_usage;
                        if !started {
                            for out in generator.generate(OutputChunk::Start { system_fingerprint, usage: prompt_usage }).await {
                                yield Ok(out);
                            }
                            started = true;
                        }
                        continue;
                    }
                    InferenceChunk::Finish { finish_reason } => {
                        if !started {
                            for out in generator.generate(OutputChunk::Start { system_fingerprint: None, usage: prompt_usage }).await {
                                yield Ok(out);
                            }
                            started = true;
                        }
                        backend_finish = Some(finish_reason);
                        break;
                    }
                    InferenceChunk::Text {
                        content,
                        content_tokens,
                    } => (content, content_tokens),
                    InferenceChunk::Token { token_id } => {
                        let Some(decoder) = decoder.as_mut() else {
                            yield Err(StreamError::MissingTokenizer);
                            return;
                        };
                        match decoder.decode(token_id) {
                            Ok(Some(decoded)) => decoded,
                            Ok(None) => continue,
                            Err(error) => {
                                yield Err(error);
                                return;
                            }
                        }
                    }
                };
                if !started {
                    for out in generator.generate(OutputChunk::Start { system_fingerprint: None, usage: prompt_usage }).await {
                        yield Ok(out);
                    }
                    started = true;
                }
                completion_usage.completion_tokens += content_tokens;
                let actions = state_machine.feed(&content);
                stashed.push(Some(content));
                for out in stashed.apply_actions(actions, &mut generator).await {
                    yield Ok(out);
                }
                stop_sequence = stashed.take_stop_sequence();
                if stop_sequence.is_some() {
                    break;
                }
            }

            if !started {
                for out in generator.generate(OutputChunk::Start { system_fingerprint: None, usage: prompt_usage }).await {
                    yield Ok(out);
                }
            }
            let actions = state_machine.finish();
            for out in stashed.apply_actions(actions, &mut generator).await {
                yield Ok(out);
            }
            // Combine the parts of a stop sequence matched across source chunks.
            if let Some(tail) = stashed.take_stop_sequence() {
                stop_sequence.get_or_insert_with(String::new).push_str(&tail);
            }
            let reason = if stop_sequence.is_some() {
                FinishReason::StopSequence
            } else {
                match backend_finish {
                    Some(InferenceFinishReason::Stop) if stashed.has_tool_calls => {
                        FinishReason::ToolCalls
                    }
                    Some(InferenceFinishReason::Stop) => FinishReason::Stop,
                    Some(InferenceFinishReason::Length) => FinishReason::Length,
                    Some(InferenceFinishReason::ContentFilter) => FinishReason::ContentFilter,
                    None => FinishReason::EndOfStream,
                }
            };
            let finish = OutputChunk::Finish {
                reason,
                stop_sequence,
                usage: completion_usage,
            };
            for out in generator.generate(finish).await {
                yield Ok(out);
            }
        }
    }
}

struct StashedChunks {
    chunks: VecDeque<Option<String>>,
    last_stashed_action: Option<OutputActionSegment>,
    stashed_tool_name: String,
    last_pop_action: OutputAction,
    stop_sequence: Option<String>,
    has_tool_calls: bool,
}

impl StashedChunks {
    fn new() -> Self {
        Self {
            chunks: VecDeque::new(),
            last_stashed_action: None,
            stashed_tool_name: String::new(),
            last_pop_action: OutputAction::Skip,
            stop_sequence: None,
            has_tool_calls: false,
        }
    }

    fn push(&mut self, content: Option<String>) {
        self.chunks.push_back(content);
    }

    fn take_stop_sequence(&mut self) -> Option<String> {
        self.stop_sequence.take()
    }

    fn pop(&mut self, action: OutputActionSegment) -> Option<OutputChunk> {
        let mut front_chunk = self.chunks.pop_front()?;
        let front_len = front_chunk
            .as_ref()
            .map(|content| content.len())
            .unwrap_or(0);
        if action.len < front_len {
            let remaining = front_chunk
                .as_mut()
                .map(|content| content.split_off(action.len));
            self.chunks.push_front(remaining);
        }
        let last_pop_action = self.last_pop_action;
        self.last_pop_action = action.action;

        match action.action {
            OutputAction::Raw => non_empty(front_chunk).map(|content| OutputChunk::Raw { content }),
            OutputAction::Reasoning => {
                non_empty(front_chunk).map(|content| OutputChunk::Reasoning { content })
            }
            OutputAction::ToSpace => Some(OutputChunk::Raw {
                content: " ".to_string(),
            }),
            OutputAction::Skip | OutputAction::SkipInvalid { .. } => None,
            OutputAction::StopSequence => {
                let content = front_chunk.unwrap_or_default();
                self.stop_sequence
                    .get_or_insert_with(String::new)
                    .push_str(&content);
                None
            }
            OutputAction::ToolCallBegin => {
                if last_pop_action != OutputAction::ToolCallBegin {
                    Some(OutputChunk::ToolCallBegin)
                } else {
                    None
                }
            }
            OutputAction::ToolName => {
                if let Some(content) = front_chunk {
                    self.stashed_tool_name.push_str(&content);
                }
                None
            }
            OutputAction::ToolNameEnd => {
                if last_pop_action != OutputAction::ToolNameEnd {
                    self.has_tool_calls = true;
                    let tool_name = std::mem::take(&mut self.stashed_tool_name);
                    Some(OutputChunk::ToolCall {
                        tool_name,
                        arguments: String::new(),
                    })
                } else {
                    None
                }
            }
            OutputAction::LabelToolCallArguments { label, .. } => {
                if last_pop_action != action.action {
                    Some(OutputChunk::ToolArgumentsDelta {
                        content: label.to_string(),
                    })
                } else {
                    None
                }
            }
            OutputAction::RawToolCallArguments { string } => {
                let content = if string {
                    escape_json_string(front_chunk.unwrap_or_default())
                } else {
                    front_chunk.unwrap_or_default()
                };
                Some(OutputChunk::ToolArgumentsDelta { content })
            }
            OutputAction::ToolCallArgumentsEnd { output } => {
                if last_pop_action == action.action {
                    return None;
                }
                output.map(|output| OutputChunk::ToolArgumentsDelta {
                    content: output.to_string(),
                })
            }
            OutputAction::Label(label) => {
                if last_pop_action != action.action {
                    Some(OutputChunk::Raw {
                        content: label.to_string(),
                    })
                } else {
                    None
                }
            }
        }
    }

    fn apply_whole_chunks(
        &mut self,
        action: &mut OutputActionSegment,
        outputs: &mut Vec<OutputChunk>,
    ) {
        loop {
            let front_len = self
                .chunks
                .front()
                .map(|content| content.as_ref().map(|content| content.len()).unwrap_or(0));
            if front_len.is_some_and(|front_len| front_len <= action.len) {
                let front_len = front_len.unwrap();
                if let Some(chunk) = self.pop(OutputActionSegment::new(action.action, front_len)) {
                    outputs.push(chunk);
                }
                action.len -= front_len;
            } else {
                return;
            }
        }
    }

    async fn apply_actions<G>(
        &mut self,
        actions: Vec<OutputActionSegment>,
        generator: &mut G,
    ) -> Vec<G::Chunk>
    where
        G: ChunkGenerator,
    {
        let mut outputs = Vec::new();
        let mut last_stashed_action = self.last_stashed_action.take();
        for action in actions {
            if let Some(last_stashed_action) = &mut last_stashed_action {
                if action.action == last_stashed_action.action {
                    last_stashed_action.len += action.len;
                } else {
                    self.apply_whole_chunks(last_stashed_action, &mut outputs);
                    if last_stashed_action.len > 0
                        && let Some(chunk) = self.pop(last_stashed_action.clone())
                    {
                        outputs.push(chunk);
                    }
                    *last_stashed_action = action;
                }
            } else {
                last_stashed_action = Some(action);
            }
        }
        if let Some(last_stashed_action) = &mut last_stashed_action {
            self.apply_whole_chunks(last_stashed_action, &mut outputs);
        }
        self.last_stashed_action = last_stashed_action;

        let mut chunks = Vec::new();
        for output in outputs {
            chunks.extend(generator.generate(output).await);
        }
        chunks
    }
}

fn non_empty(content: Option<String>) -> Option<String> {
    content.filter(|content| !content.is_empty())
}

fn escape_json_string(content: String) -> String {
    match serde_json::to_string(&content) {
        Ok(escaped) if escaped.len() > 2 => escaped[1..escaped.len() - 1].to_string(),
        _ => String::new(),
    }
}
