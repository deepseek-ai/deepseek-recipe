/// Decode the leading JSON `input` string, preserving nonmatching arguments.
/// Content after the closing quote is ignored.
#[derive(Debug, Default)]
pub(super) struct ResponsesCustomToolInputParser {
    state: ResponsesCustomToolInputParserState,
    prefix: String,
    prefix_part: usize,
    prefix_offset: usize,
    escape: String,
}

#[derive(Debug, Default)]
enum ResponsesCustomToolInputParserState {
    #[default]
    Prefix,
    Value,
    Passthrough,
    Done,
}

const PREFIX_PARTS: [&[u8]; 4] = [b"{", b"\"input\"", b":", b"\""];

impl ResponsesCustomToolInputParser {
    /// Decode available value text, retaining incomplete prefixes and escapes.
    pub(super) fn feed(&mut self, chunk: &str) -> String {
        let mut output = String::new();
        for ch in chunk.chars() {
            match self.state {
                ResponsesCustomToolInputParserState::Prefix => self.feed_prefix(ch, &mut output),
                ResponsesCustomToolInputParserState::Value => self.feed_value(ch, &mut output),
                ResponsesCustomToolInputParserState::Passthrough => output.push(ch),
                ResponsesCustomToolInputParserState::Done => break,
            }
        }
        output
    }

    /// End parsing and return any unfinished prefix or escape unchanged.
    /// Repeated calls and subsequent input produce no further output.
    pub(super) fn finish(&mut self) -> String {
        self.state = ResponsesCustomToolInputParserState::Done;
        let mut output = std::mem::take(&mut self.prefix);
        output.push_str(&std::mem::take(&mut self.escape));
        output
    }

    fn feed_prefix(&mut self, ch: char, output: &mut String) {
        self.prefix.push(ch);
        let part = PREFIX_PARTS[self.prefix_part];
        if ch == char::from(part[self.prefix_offset]) {
            self.prefix_offset += 1;
            if self.prefix_offset == part.len() {
                self.prefix_part += 1;
                self.prefix_offset = 0;
                if self.prefix_part == PREFIX_PARTS.len() {
                    self.prefix.clear();
                    self.state = ResponsesCustomToolInputParserState::Value;
                }
            }
        } else if self.prefix_offset != 0 || !ch.is_ascii_whitespace() {
            output.push_str(&self.prefix);
            self.prefix.clear();
            self.state = ResponsesCustomToolInputParserState::Passthrough;
        }
    }

    fn feed_value(&mut self, ch: char, output: &mut String) {
        if self.escape.is_empty() {
            match ch {
                '\\' => self.escape.push(ch),
                '"' => self.state = ResponsesCustomToolInputParserState::Done,
                _ => output.push(ch),
            }
            return;
        }
        if ch == '"' && !self.escape.ends_with('\\') {
            output.push_str(&std::mem::take(&mut self.escape));
            self.state = ResponsesCustomToolInputParserState::Done;
            return;
        }
        self.escape.push(ch);
        if incomplete_escape(&self.escape) {
            return;
        }
        let raw = std::mem::take(&mut self.escape);
        match serde_json::from_str::<String>(&format!("\"{raw}\"")) {
            Ok(decoded) => output.push_str(&decoded),
            Err(_) => output.push_str(&raw),
        }
    }
}

fn incomplete_escape(raw: &str) -> bool {
    if raw.len() < 6 {
        return partial_unicode_escape(raw);
    }
    // Validate the first ASCII escape before inspecting a surrogate pair.
    // Invalid escapes can contain multibyte characters at any position.
    let Some(first) = raw.get(..6) else {
        return false;
    };
    if !partial_unicode_escape(first) {
        return false;
    }
    let high_surrogate =
        u16::from_str_radix(&first[2..], 16).is_ok_and(|value| (0xD800..=0xDBFF).contains(&value));
    high_surrogate && raw.len() < 12 && partial_unicode_escape(&raw[6..])
}

fn partial_unicode_escape(raw: &str) -> bool {
    raw.len() <= 6
        && ("\\u".starts_with(raw)
            || raw
                .strip_prefix("\\u")
                .is_some_and(|digits| digits.chars().all(|ch| ch.is_ascii_hexdigit())))
}
