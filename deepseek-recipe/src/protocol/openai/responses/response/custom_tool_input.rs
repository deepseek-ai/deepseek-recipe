/// A streaming decoder of the leading JSON `input` string of tool call
/// arguments that passes nonmatching arguments through unchanged. Content
/// after the closing quote is ignored.
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
        let mut raw = std::mem::take(&mut self.escape);
        raw.push(ch);
        self.feed_escape(raw, output);
    }

    /// Decode a buffered escape, keep buffering an incomplete unicode escape,
    /// or write an invalid escape as raw text.
    fn feed_escape(&mut self, raw: String, output: &mut String) {
        // A two-character escape such as `\n` is complete; `\u` starts a
        // six-character unicode escape.
        if raw.len() == 2 {
            if raw == "\\u" {
                self.escape = raw;
            } else {
                push_decoded_escape(output, &raw);
            }
            return;
        }
        if !raw.starts_with("\\u") {
            output.push_str(&raw);
            return;
        }
        if raw.len() < 6 {
            if is_partial_unicode_escape(&raw) {
                self.escape = raw;
            } else {
                output.push_str(&raw);
            }
            return;
        }
        // A high surrogate waits for the low surrogate of its pair.
        if raw.len() == 6 {
            if !is_partial_unicode_escape(&raw) {
                output.push_str(&raw);
            } else if is_high_surrogate_escape(&raw) {
                self.escape = raw;
            } else {
                push_decoded_escape(output, &raw);
            }
            return;
        }
        if raw.len() < 12 {
            if raw.get(6..).is_some_and(is_partial_unicode_escape) {
                self.escape = raw;
            } else {
                output.push_str(&raw);
            }
            return;
        }
        if raw.len() == 12 && raw.get(6..).is_some_and(is_low_surrogate_escape) {
            push_decoded_escape(output, &raw);
        } else {
            output.push_str(&raw);
        }
    }
}

/// Write the decoded text of a JSON escape, or the raw escape when it does not
/// decode.
fn push_decoded_escape(output: &mut String, raw: &str) {
    match serde_json::from_str::<String>(&format!("\"{raw}\"")) {
        Ok(decoded) => output.push_str(&decoded),
        Err(_) => output.push_str(raw),
    }
}

/// Return whether a `\uXXXX` escape encodes a code unit in `range`.
fn is_surrogate_escape_in(raw: &str, range: std::ops::RangeInclusive<u32>) -> bool {
    raw.get(2..)
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .is_some_and(|code_unit| range.contains(&code_unit))
}

/// Return whether a `\uXXXX` escape encodes a high surrogate.
fn is_high_surrogate_escape(raw: &str) -> bool {
    is_surrogate_escape_in(raw, 0xD800..=0xDBFF)
}

/// Return whether a `\uXXXX` escape encodes a low surrogate.
fn is_low_surrogate_escape(raw: &str) -> bool {
    is_surrogate_escape_in(raw, 0xDC00..=0xDFFF)
}

/// Return whether `raw` is a prefix of a `\uXXXX` escape, including the
/// complete escape.
fn is_partial_unicode_escape(raw: &str) -> bool {
    raw.len() <= 6
        && ("\\u".starts_with(raw)
            || raw
                .strip_prefix("\\u")
                .is_some_and(|digits| digits.chars().all(|ch| ch.is_ascii_hexdigit())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(chunks: &[&str]) -> String {
        let mut parser = ResponsesCustomToolInputParser::default();
        let mut output: String = chunks.iter().map(|chunk| parser.feed(chunk)).collect();
        output.push_str(&parser.finish());
        output
    }

    #[test]
    fn escapes_decode_across_chunks() {
        assert_eq!(
            parse(&[r#"{"input": "a\"#, r#"nb\u00"#, r#"e9"}"#]),
            "a\nbé"
        );
        assert_eq!(parse(&[r#"{"input":"\uD83D"#, r#"\uDE00"}"#]), "😀");
    }

    #[test]
    fn invalid_escapes_stay_raw_and_the_value_continues() {
        // A quote inside an unfinished unicode escape is part of the raw text.
        assert_eq!(parse(&[r#"{"input": "\u12"x"}"#]), r#"\u12"x"#);
        assert_eq!(parse(&[r#"{"input": "\uD83Dz"}"#]), r#"\uD83Dz"#);
    }

    #[test]
    fn other_arguments_pass_through() {
        assert_eq!(parse(&[r#"{"other": 1}"#]), r#"{"other": 1}"#);
        assert_eq!(parse(&[r#"{"inp"#]), r#"{"inp"#);
    }
}
