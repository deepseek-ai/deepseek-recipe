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
/// An ASCII `id` carries the marker when its bytes at zero-based offsets 8
/// to 10 are `ET_`, the position in IDs from [`new_tool_call_id`], or, for an
/// ID whose underscores were removed (`call00ET…`), when its bytes at offsets
/// 6 and 7 are `ET`.
pub(crate) fn is_empty_thinking_tool_call_id(id: &str) -> bool {
    id.is_ascii() && ((id.len() > 10 && &id[8..11] == "ET_") || (id.len() > 7 && &id[6..8] == "ET"))
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
