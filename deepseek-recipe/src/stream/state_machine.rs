//! Incrementally recognize reasoning, tool-call markup, and stop sequences.

const JSON_BEGIN_LABELS: [&str; 2] = ["```json\n", "```\n"];

/// Position within the reasoning section when parsing begins.
#[derive(Debug, Clone, Copy)]
pub enum ReasoningStage {
    /// Begin reasoning while discarding leading newlines.
    Start,
    /// Continue reasoning while preserving leading newlines.
    Reasoning,
    /// Continue answer content while preserving leading newlines.
    Content,
}

impl ReasoningStage {
    pub fn start_from_reasoning(&self) -> bool {
        matches!(self, Self::Start | Self::Reasoning)
    }
}

const RAW_JSON_BEGIN_LABELS: [&str; 2] = ["{", "["];
const JSON_END_LABEL: &str = "\n```";
const DSML_BEGIN_LABEL: &str = "<｜DSML｜";
const DSML_END_LABEL: &str = "</｜DSML｜";
const DSML_TOOL_CALLS_END_LABELS: [&str; 2] = ["</｜DSML｜tool_calls>", "</｜DSML｜ calls>"];
pub const REASONING_END_LABEL: &str = "</think>";

/// Settings for parsing inference output.
#[derive(Debug, Clone)]
pub struct ParsingOptions {
    /// Parse model-specific tool-call markup.
    pub parse_tool_calls: bool,
    /// Begin parsing inside the tool-call block specified by the prompt.
    /// Takes precedence over `reasoning_initial_stage`.
    pub tool_call_initial_stage: bool,
    /// Recognize JSON fences or a raw `{` or `[` prefix and replace surrounding
    /// text with spaces. This does not validate JSON syntax.
    pub parse_json_output: bool,
    /// Initial reasoning or answer stage. `None` starts answer content with
    /// leading-newline removal.
    pub reasoning_initial_stage: Option<ReasoningStage>,
    /// Match stop sequences in ordinary answer text and detected JSON output.
    /// Reasoning, tool-call markup, and surrounding JSON-mode text are excluded.
    /// The state machine discards empty sequences during construction.
    pub stop_sequences: Vec<String>,
}

impl Default for ParsingOptions {
    fn default() -> Self {
        Self {
            parse_tool_calls: true,
            tool_call_initial_stage: false,
            parse_json_output: false,
            reasoning_initial_stage: None,
            stop_sequences: Vec::new(),
        }
    }
}

/// An incremental parser whose output segments refer to input byte lengths.
pub struct StateMachine {
    options: ParsingOptions,
    state: State,
}

impl StateMachine {
    pub fn new(mut options: ParsingOptions) -> Self {
        options
            .stop_sequences
            .retain(|sequence| !sequence.is_empty());
        let stage = if options.tool_call_initial_stage {
            Stage::ToolCalls
        } else if let Some(reasoning_initial_stage) = &options.reasoning_initial_stage {
            match reasoning_initial_stage {
                ReasoningStage::Start => Stage::Reasoning { is_leading: true },
                ReasoningStage::Reasoning => Stage::Reasoning { is_leading: false },
                ReasoningStage::Content => Stage::Common { is_leading: false },
            }
        } else {
            Stage::Common { is_leading: true }
        };
        let state = State::new(stage, &options);
        Self { options, state }
    }

    /// Parse the next source chunk, retaining incomplete markers for later input.
    pub fn feed(&mut self, content: &str) -> Vec<OutputActionSegment> {
        self.state.feed(content, &self.options)
    }

    /// Process buffered input when the source ends.
    pub fn finish(self) -> Vec<OutputActionSegment> {
        self.state.finish().1
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InvalidTokenKind {
    ExtraEndOfThinking,
    ContentAfterFinished,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputAction {
    Raw,
    ToSpace,
    SkipInvalid { kind: InvalidTokenKind },
    Skip,
    StopSequence,
    ToolCallBegin,
    ToolName,
    ToolNameEnd,
    LabelToolCallArguments { label: &'static str, id: u32 },
    RawToolCallArguments { string: bool },
    ToolCallArgumentsEnd { output: Option<&'static str> },
    Reasoning,
    Label(&'static str),
}

#[derive(Debug, Clone)]
pub struct OutputActionSegment {
    pub action: OutputAction,
    pub len: usize,
}

impl OutputActionSegment {
    pub fn new(action: OutputAction, len: usize) -> Self {
        Self { action, len }
    }
}

#[derive(Debug, Clone, Copy)]
enum Stage {
    Common { is_leading: bool },
    Json,
    MatchedJson,
    ToolCalls,
    ToolName,
    ToolCallArguments { is_leading: bool },
    ToolCallParamName,
    ToolCallParamType,
    ToolCallParamValue { string: bool },
    Reasoning { is_leading: bool },
    Finished,
}

#[derive(Debug)]
struct MatchBranch {
    pub state: MatchState,
    pub next_stage: Stage,
    pub action_on_matched: ActionOnMatched,
}

#[derive(Debug)]
struct MatchBranches(Vec<MatchBranch>);

impl MatchBranches {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn new_branch(
        &mut self,
        pattern: MatchPattern,
        next_stage: Stage,
        action_on_matched: ActionOnMatched,
    ) {
        self.0.push(MatchBranch {
            state: pattern.new_state(),
            next_stage,
            action_on_matched,
        });
    }
    fn new_string_branch<T: ToString + ?Sized>(
        &mut self,
        label: &T,
        next_stage: Stage,
        action_on_matched: ActionOnMatched,
    ) {
        self.new_branch(
            MatchPattern::String(label.to_string()),
            next_stage,
            action_on_matched,
        );
    }
    fn new_strings_branch<T: ToString>(
        &mut self,
        labels: &[T],
        next_stage: Stage,
        action_on_matched: ActionOnMatched,
    ) {
        labels
            .iter()
            .for_each(|label| self.new_string_branch(label, next_stage, action_on_matched));
    }
    fn add_tool_call_begin_branches(&mut self) {
        self.new_branch(
            MatchPattern::NewlinesAndString(DSML_BEGIN_LABEL.to_string()),
            Stage::ToolCalls,
            ActionOnMatched::Skip,
        );
    }
}

#[derive(Debug)]
struct State {
    stage: Stage,
    match_branches: MatchBranches,
    action_on_unmatched: ActionOnUnmatched,
    stashed_size: usize,
}

type ActionOnMatched = OutputAction;
type ActionOnUnmatched = OutputAction;

impl State {
    fn new(stage: Stage, options: &ParsingOptions) -> Self {
        let mut match_branches = MatchBranches::new();
        let mut add_stop_sequence_branches = || {
            match_branches.new_strings_branch(
                &options.stop_sequences,
                Stage::Finished,
                ActionOnMatched::StopSequence,
            );
        };
        let action_on_unmatched = match stage {
            Stage::Common { is_leading } => {
                if options.parse_json_output {
                    match_branches.new_strings_branch(
                        &JSON_BEGIN_LABELS,
                        Stage::Json,
                        ActionOnMatched::Skip,
                    );
                    for raw_json_label in RAW_JSON_BEGIN_LABELS {
                        match_branches.new_string_branch(
                            raw_json_label,
                            Stage::Json,
                            ActionOnMatched::Label(raw_json_label),
                        );
                    }
                } else {
                    add_stop_sequence_branches();
                }
                if options.parse_tool_calls {
                    match_branches.add_tool_call_begin_branches();
                }
                if is_leading {
                    match_branches.new_branch(
                        MatchPattern::LeadingNewline,
                        Stage::Common { is_leading: true },
                        ActionOnMatched::Skip,
                    );
                }
                match_branches.new_string_branch(
                    REASONING_END_LABEL,
                    Stage::Common { is_leading: false },
                    ActionOnMatched::SkipInvalid {
                        kind: InvalidTokenKind::ExtraEndOfThinking,
                    },
                );
                if options.parse_json_output {
                    ActionOnUnmatched::ToSpace
                } else {
                    ActionOnUnmatched::Raw
                }
            }
            Stage::Json => {
                add_stop_sequence_branches();
                match_branches.new_string_branch(
                    JSON_END_LABEL,
                    Stage::MatchedJson,
                    ActionOnMatched::Skip,
                );
                ActionOnUnmatched::Raw
            }
            Stage::MatchedJson => {
                if options.parse_tool_calls {
                    match_branches.add_tool_call_begin_branches();
                }
                ActionOnUnmatched::ToSpace
            }
            Stage::ToolCalls => {
                match_branches.new_string_branch(
                    "invoke name=\"",
                    Stage::ToolName,
                    ActionOnMatched::ToolCallBegin,
                );
                match_branches.new_strings_branch(
                    &DSML_TOOL_CALLS_END_LABELS,
                    Stage::Finished,
                    ActionOnMatched::Skip,
                );
                ActionOnUnmatched::Skip
            }
            Stage::ToolName => {
                match_branches.new_string_branch(
                    "\"",
                    Stage::ToolCallArguments { is_leading: true },
                    ActionOnMatched::ToolNameEnd,
                );
                ActionOnUnmatched::ToolName
            }
            Stage::ToolCallArguments { is_leading } => {
                match_branches.new_string_branch(
                    "parameter name=\"",
                    Stage::ToolCallParamName,
                    if is_leading {
                        ActionOnMatched::LabelToolCallArguments {
                            label: "{\"",
                            id: 0,
                        }
                    } else {
                        ActionOnMatched::LabelToolCallArguments {
                            label: ", \"",
                            id: 1,
                        }
                    },
                );
                match_branches.new_string_branch(
                    DSML_END_LABEL,
                    Stage::ToolCalls,
                    if is_leading {
                        ActionOnMatched::ToolCallArgumentsEnd { output: Some("{}") }
                    } else {
                        ActionOnMatched::ToolCallArgumentsEnd { output: Some("}") }
                    },
                );
                ActionOnUnmatched::Skip
            }
            Stage::ToolCallParamName => {
                match_branches.new_string_branch(
                    "\"",
                    Stage::ToolCallParamType,
                    ActionOnMatched::LabelToolCallArguments {
                        label: "\": ",
                        id: 2,
                    },
                );
                ActionOnUnmatched::RawToolCallArguments { string: false }
            }
            Stage::ToolCallParamType => {
                match_branches.new_string_branch(
                    "true\">",
                    Stage::ToolCallParamValue { string: true },
                    ActionOnMatched::LabelToolCallArguments { label: "\"", id: 3 },
                );
                match_branches.new_string_branch(
                    "false\">",
                    Stage::ToolCallParamValue { string: false },
                    ActionOnMatched::Skip,
                );
                ActionOnUnmatched::Skip
            }
            Stage::ToolCallParamValue { string } => {
                match_branches.new_string_branch(
                    DSML_END_LABEL,
                    Stage::ToolCallArguments { is_leading: false },
                    if string {
                        ActionOnMatched::LabelToolCallArguments { label: "\"", id: 4 }
                    } else {
                        ActionOnMatched::Skip
                    },
                );
                ActionOnUnmatched::RawToolCallArguments { string }
            }
            Stage::Reasoning { is_leading } => {
                if options.parse_tool_calls {
                    match_branches.add_tool_call_begin_branches();
                }
                if is_leading {
                    match_branches.new_branch(
                        MatchPattern::LeadingNewline,
                        Stage::Reasoning { is_leading },
                        ActionOnMatched::Skip,
                    );
                }
                match_branches.new_branch(
                    MatchPattern::NewlinesAndString(REASONING_END_LABEL.to_string()),
                    Stage::Common { is_leading: true },
                    ActionOnMatched::Skip,
                );
                ActionOnUnmatched::Reasoning
            }
            Stage::Finished => ActionOnUnmatched::SkipInvalid {
                kind: InvalidTokenKind::ContentAfterFinished,
            },
        };
        Self {
            stage,
            match_branches,
            action_on_unmatched,
            stashed_size: 0,
        }
    }

    fn feed(&mut self, content: &str, options: &ParsingOptions) -> Vec<OutputActionSegment> {
        let mut output_actions = Vec::new();
        for byte in content.bytes() {
            self.stashed_size += 1;
            let mut matched_branch = None;
            for branch in self.match_branches.0.iter_mut() {
                if branch.state.feed(byte) {
                    matched_branch = Some((
                        branch.state.matching_len(),
                        branch.next_stage,
                        branch.action_on_matched,
                    ));
                    break;
                }
            }
            if let Some((matching_len, next_stage, action_on_matched)) = matched_branch {
                if matching_len < self.stashed_size {
                    output_actions.push(OutputActionSegment::new(
                        self.action_on_unmatched,
                        self.stashed_size - matching_len,
                    ));
                }
                if matching_len > 0 {
                    output_actions.push(OutputActionSegment::new(action_on_matched, matching_len));
                }
                *self = State::new(next_stage, options);
            }
        }
        let matching_len = self
            .match_branches
            .0
            .iter()
            .map(|branch| branch.state.matching_len())
            .max()
            .unwrap_or(0);
        if matching_len < self.stashed_size {
            output_actions.push(OutputActionSegment::new(
                self.action_on_unmatched,
                self.stashed_size - matching_len,
            ));
            self.stashed_size = matching_len;
        }
        output_actions
    }

    fn finish(self) -> (Stage, Vec<OutputActionSegment>) {
        let mut output_actions = Vec::new();
        if self.stashed_size > 0 {
            output_actions.push(OutputActionSegment::new(
                self.action_on_unmatched,
                self.stashed_size,
            ));
        }

        (self.stage, output_actions)
    }
}

enum MatchPattern {
    String(String),
    NewlinesAndString(String),
    LeadingNewline,
}

impl MatchPattern {
    fn new_state(self) -> MatchState {
        match self {
            Self::String(s) => {
                let kmp_table = kmp_table(s.as_bytes());
                MatchState::String {
                    s,
                    kmp_table,
                    len: 0,
                }
            }
            Self::NewlinesAndString(s) => {
                let kmp_table = kmp_table(s.as_bytes());
                MatchState::NewlinesAndString {
                    s,
                    kmp_table,
                    len: 0,
                    newline_count: 0,
                }
            }
            Self::LeadingNewline => MatchState::LeadingNewline {
                leading: true,
                matched: false,
            },
        }
    }
}

type KMPTable = Vec<usize>;

fn kmp_table(pattern: &[u8]) -> KMPTable {
    let mut table = vec![0; pattern.len()];
    let mut len = 0;
    let mut i = 1;
    while i < pattern.len() {
        if pattern[i] == pattern[len] {
            len += 1;
            table[i] = len;
            i += 1;
        } else if len != 0 {
            len = table[len - 1];
        } else {
            table[i] = 0;
            i += 1;
        }
    }
    table
}

#[derive(Debug)]
enum MatchState {
    String {
        s: String,
        kmp_table: KMPTable,
        len: usize,
    },
    NewlinesAndString {
        s: String,
        kmp_table: KMPTable,
        len: usize,
        newline_count: usize,
    },
    LeadingNewline {
        leading: bool,
        matched: bool,
    },
}

fn kmp_next(s: &str, kmp_table: &[usize], mut len: usize, byte: u8) -> usize {
    while len > 0 && s.as_bytes().get(len) != Some(&byte) {
        len = kmp_table[len - 1];
    }
    if s.as_bytes().get(len) == Some(&byte) {
        len += 1;
    }
    len
}

impl MatchState {
    fn feed(&mut self, byte: u8) -> bool {
        match self {
            MatchState::String { s, kmp_table, len } => {
                *len = kmp_next(s, kmp_table, *len, byte);
                *len == s.len()
            }
            MatchState::NewlinesAndString {
                s,
                kmp_table,
                len,
                newline_count,
            } => {
                if *len == 0 && byte == b'\n' {
                    *newline_count += 1;
                } else {
                    let next = kmp_next(s, kmp_table, *len, byte);
                    if next <= *len {
                        *newline_count = 0
                    }
                    *len = next;
                }
                *len == s.len()
            }
            MatchState::LeadingNewline { leading, matched } => {
                if *leading {
                    *matched = byte == b'\n';
                }
                *leading = false;
                *matched
            }
        }
    }
    fn matching_len(&self) -> usize {
        match self {
            MatchState::String { len, .. } => *len,
            MatchState::NewlinesAndString {
                len, newline_count, ..
            } => *len + *newline_count,
            MatchState::LeadingNewline { matched, .. } => {
                if *matched {
                    1
                } else {
                    0
                }
            }
        }
    }
}
