use crate::util::append_delta::AppendDelta;

use super::schema::{
    ResponsesContentPart, ResponsesOutputItem, ResponsesResponse, ResponsesStreamEvent,
};

/// Apply events in generation order.
impl AppendDelta<ResponsesStreamEvent> for ResponsesResponse {
    fn append(&mut self, event: ResponsesStreamEvent) {
        match event {
            ResponsesStreamEvent::Created { response, .. }
            | ResponsesStreamEvent::InProgress { response, .. }
            | ResponsesStreamEvent::Completed { response, .. }
            | ResponsesStreamEvent::Incomplete { response, .. } => *self = response,
            ResponsesStreamEvent::OutputItemAdded {
                item, output_index, ..
            } => {
                if output_index == self.output.len() && valid_item_content(&item) {
                    self.output.push(item);
                }
            }
            ResponsesStreamEvent::OutputItemDone {
                item, output_index, ..
            } => {
                if let Some(current) = self.output.get_mut(output_index)
                    && item_id(current) == item_id(&item)
                    && std::mem::discriminant(current) == std::mem::discriminant(&item)
                    && valid_item_content(&item)
                {
                    *current = item;
                }
            }
            ResponsesStreamEvent::ContentPartAdded {
                content_index,
                item_id,
                output_index,
                part,
                ..
            } => {
                if let Some(item) = self.matching_item_mut(output_index, &item_id)
                    && let Some(content) = matching_content_mut(item, &part)
                    && content_index == content.len()
                {
                    content.push(part);
                }
            }
            ResponsesStreamEvent::ContentPartDone {
                content_index,
                item_id,
                output_index,
                part,
                ..
            } => {
                if let Some(item) = self.matching_item_mut(output_index, &item_id)
                    && let Some(content) = matching_content_mut(item, &part)
                    && let Some(current) = content.get_mut(content_index)
                    && std::mem::discriminant(current) == std::mem::discriminant(&part)
                {
                    *current = part;
                }
            }
            ResponsesStreamEvent::OutputTextDelta {
                content_index,
                delta,
                item_id,
                logprobs: delta_logprobs,
                output_index,
                ..
            } => {
                if let Some(ResponsesOutputItem::Message { content, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                    && let Some(ResponsesContentPart::OutputText { text, logprobs, .. }) =
                        content.get_mut(content_index)
                {
                    text.push_str(&delta);
                    logprobs.extend(delta_logprobs);
                }
            }
            ResponsesStreamEvent::OutputTextDone {
                content_index,
                item_id,
                logprobs: final_logprobs,
                output_index,
                text: final_text,
                ..
            } => {
                if let Some(ResponsesOutputItem::Message { content, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                    && let Some(ResponsesContentPart::OutputText { text, logprobs, .. }) =
                        content.get_mut(content_index)
                {
                    *text = final_text;
                    *logprobs = final_logprobs;
                }
            }
            ResponsesStreamEvent::ReasoningTextDelta {
                content_index,
                delta,
                item_id,
                output_index,
                ..
            } => {
                if let Some(ResponsesOutputItem::Reasoning { content, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                    && let Some(ResponsesContentPart::ReasoningText { text }) =
                        content.get_mut(content_index)
                {
                    text.push_str(&delta);
                }
            }
            ResponsesStreamEvent::ReasoningTextDone {
                content_index,
                item_id,
                output_index,
                text: final_text,
                ..
            } => {
                if let Some(ResponsesOutputItem::Reasoning { content, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                    && let Some(ResponsesContentPart::ReasoningText { text }) =
                        content.get_mut(content_index)
                {
                    *text = final_text;
                }
            }
            ResponsesStreamEvent::FunctionCallArgumentsDelta {
                delta,
                item_id,
                output_index,
                ..
            } => {
                if let Some(ResponsesOutputItem::FunctionCall { arguments, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                {
                    arguments.push_str(&delta);
                }
            }
            ResponsesStreamEvent::FunctionCallArgumentsDone {
                arguments: final_arguments,
                item_id,
                output_index,
                ..
            } => {
                if let Some(ResponsesOutputItem::FunctionCall { arguments, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                {
                    *arguments = final_arguments;
                }
            }
            ResponsesStreamEvent::CustomToolCallInputDelta {
                delta,
                item_id,
                output_index,
                ..
            } => {
                if let Some(ResponsesOutputItem::CustomToolCall { input, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                {
                    input.push_str(&delta);
                }
            }
            ResponsesStreamEvent::CustomToolCallInputDone {
                input: final_input,
                item_id,
                output_index,
                ..
            } => {
                if let Some(ResponsesOutputItem::CustomToolCall { input, .. }) =
                    self.matching_item_mut(output_index, &item_id)
                {
                    *input = final_input;
                }
            }
        }
    }
}

impl ResponsesResponse {
    fn matching_item_mut(
        &mut self,
        output_index: usize,
        expected_id: &str,
    ) -> Option<&mut ResponsesOutputItem> {
        self.output
            .get_mut(output_index)
            .filter(|item| item_id(item) == expected_id)
    }
}

fn item_id(item: &ResponsesOutputItem) -> &str {
    match item {
        ResponsesOutputItem::Message { id, .. }
        | ResponsesOutputItem::Reasoning { id, .. }
        | ResponsesOutputItem::FunctionCall { id, .. }
        | ResponsesOutputItem::CustomToolCall { id, .. } => id,
    }
}

fn valid_item_content(item: &ResponsesOutputItem) -> bool {
    match item {
        ResponsesOutputItem::Message { content, .. } => content
            .iter()
            .all(|part| matches!(part, ResponsesContentPart::OutputText { .. })),
        ResponsesOutputItem::Reasoning { content, .. } => content
            .iter()
            .all(|part| matches!(part, ResponsesContentPart::ReasoningText { .. })),
        ResponsesOutputItem::FunctionCall { .. } | ResponsesOutputItem::CustomToolCall { .. } => {
            true
        }
    }
}

fn matching_content_mut<'a>(
    item: &'a mut ResponsesOutputItem,
    part: &ResponsesContentPart,
) -> Option<&'a mut Vec<ResponsesContentPart>> {
    match (item, part) {
        (ResponsesOutputItem::Message { content, .. }, ResponsesContentPart::OutputText { .. })
        | (
            ResponsesOutputItem::Reasoning { content, .. },
            ResponsesContentPart::ReasoningText { .. },
        ) => Some(content),
        _ => None,
    }
}
