//! Tool call ID generation and inspection.
//!
//! An ID is `call_{index:02}_` followed by an optional `ET_` marker, 20 random
//! alphanumeric characters, and a four-digit checksum. For thinking-mode rounds
//! that produced no reasoning content, IDs carry the `ET_` marker. The marker
//! lets conversion of a later request distinguish an assistant turn in which
//! the model produced no reasoning content from a historical assistant message
//! whose reasoning content the request omits.

use rand::Rng;
use rand::distr::Alphanumeric;
use xxhash_rust::xxh3::xxh3_64;

/// Return the checksum of an ID without its checksum: the XXH3-64 hash of the
/// ID with underscores removed, modulo 10000.
fn tool_call_id_checksum(id_without_checksum: &str) -> u64 {
    let input: String = id_without_checksum
        .chars()
        .filter(|ch| *ch != '_')
        .collect();
    xxh3_64(input.as_bytes()) % 10000
}

/// Generate a tool call ID for `index`, marking rounds with empty thinking.
pub(crate) fn new_tool_call_id(index: usize, empty_thinking: bool) -> String {
    let random_part: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(20)
        .map(char::from)
        .collect();
    let empty_thinking_mark = if empty_thinking { "ET_" } else { "" };
    let id_without_checksum = format!("call_{index:02}_{empty_thinking_mark}{random_part}");
    let checksum = tool_call_id_checksum(&id_without_checksum);
    format!("{id_without_checksum}{checksum:04}")
}

/// Return whether `id` carries the empty-thinking marker.
///
/// The marker follows the complete decimal index, whose width is at least
/// two digits. IDs whose underscores were removed are accepted as well.
pub(crate) fn is_empty_thinking_tool_call_id(id: &str) -> bool {
    if !id.is_ascii() {
        return false;
    }
    if let Some(rest) = id.strip_prefix("call_") {
        return rest.split_once('_').is_some_and(|(index, tail)| {
            index.len() >= 2
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && tail.starts_with("ET_")
        });
    }
    id.strip_prefix("call").is_some_and(|rest| {
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        digits >= 2 && rest[digits..].starts_with("ET")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_ids_encode_the_empty_thinking_mark() {
        assert!(!is_empty_thinking_tool_call_id(&new_tool_call_id(0, false)));
        assert!(is_empty_thinking_tool_call_id(&new_tool_call_id(0, true)));
    }

    #[test]
    fn generated_ids_end_with_their_checksum() {
        let id = new_tool_call_id(3, true);
        assert_eq!(id.len(), 35);
        let (id_without_checksum, checksum) = id.split_at(id.len() - 4);
        assert_eq!(
            checksum,
            format!("{:04}", tool_call_id_checksum(id_without_checksum))
        );
        assert!(id.starts_with("call_03_ET_"));
    }

    #[test]
    fn foreign_ids_are_not_marked() {
        assert!(!is_empty_thinking_tool_call_id("toolu_123"));
        assert!(!is_empty_thinking_tool_call_id(""));
    }
}

#[cfg(test)]
mod contribution_regression {
    use super::*;

    #[test]
    fn generated_large_indexes_keep_the_marker() {
        for index in [0, 99, 100, 1000, usize::MAX] {
            let id = new_tool_call_id(index, true);
            assert!(is_empty_thinking_tool_call_id(&id), "{id}");
            assert!(is_empty_thinking_tool_call_id(&id.replace('_', "")), "{id}");
            assert!(!is_empty_thinking_tool_call_id(&new_tool_call_id(
                index, false
            )));
        }
    }
}
