use std::collections::{HashMap, HashSet};

use deepseek_recipe_core::messages::ToolCall;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::protocol::validation::{validate_tool_name, validate_tool_parameters};
use crate::request::{ConversionError, ConversionOptions, WebSearchBehavior};

use super::schema::{
    ResponsesFunctionCall, ResponsesFunctionTool, ResponsesNamedToolChoice, ResponsesNamespaceTool,
    ResponsesTool, ResponsesToolChoice, ResponsesToolChoiceMode,
};

pub(super) struct ResponsesToolSet {
    definitions: Vec<ResponsesFunctionTool>,
    namespace_names: HashMap<(String, String), String>,
    top_level_names: HashSet<String>,
}

impl ResponsesToolSet {
    pub(super) fn convert(
        tools: Option<Vec<ResponsesTool>>,
        options: &ConversionOptions,
    ) -> Result<Self, ConversionError> {
        let tools = tools.unwrap_or_default();
        let mut converted = Self {
            definitions: Vec::new(),
            namespace_names: HashMap::new(),
            top_level_names: tools
                .iter()
                .filter_map(|tool| match tool {
                    ResponsesTool::Function(function) => Some(function.name.clone()),
                    _ => None,
                })
                .collect(),
        };
        let mut namespaces = HashSet::new();
        for (index, tool) in tools.into_iter().enumerate() {
            let path = format!("tools[{index}]");
            match tool {
                ResponsesTool::Function(function) => {
                    validate_tool_name(&function.name, &format!("{path}.name"))?;
                    converted.definitions.push(function);
                }
                ResponsesTool::Namespace {
                    name,
                    description,
                    tools,
                } => {
                    validate_tool_name(&name, &format!("{path}.name"))?;
                    if !namespaces.insert(name.clone()) {
                        return Err(ConversionError::bad_request(format!(
                            "{path}: duplicate namespace name '{name}'"
                        )));
                    }
                    let mut inner_names = HashSet::new();
                    for (inner_index, tool) in tools.into_iter().enumerate() {
                        let inner_path = format!("{path}.tools[{inner_index}]");
                        let ResponsesNamespaceTool::Function(mut function) = tool else {
                            return Err(ConversionError::bad_request(format!(
                                "{inner_path}: custom tools are not supported inside a namespace"
                            )));
                        };
                        validate_tool_name(&function.name, &format!("{inner_path}.name"))?;
                        if !inner_names.insert(function.name.clone()) {
                            return Err(ConversionError::bad_request(format!(
                                "{inner_path}: tool names within a namespace must be unique"
                            )));
                        }
                        if let Some(namespace_description) = description
                            .as_deref()
                            .filter(|description| !description.is_empty())
                        {
                            let function_description =
                                function.description.as_deref().unwrap_or_default();
                            function.description =
                                Some(format!("{namespace_description}\n{function_description}"));
                        }
                        let function_name =
                            unique_name(join_namespace(&name, &function.name), |candidate| {
                                converted.is_name_taken(candidate)
                            });
                        converted
                            .namespace_names
                            .insert((name.clone(), function.name), function_name.clone());
                        function.name = function_name;
                        converted.definitions.push(function);
                    }
                }
                ResponsesTool::Custom { name } => {
                    if name != "apply_patch" {
                        return Err(ConversionError::bad_request(format!(
                            "Unsupported custom tool: '{name}'. Only 'apply_patch' is supported"
                        )));
                    }
                    converted.definitions.push(apply_patch_tool());
                }
                ResponsesTool::WebSearch => match options.responses_web_search {
                    WebSearchBehavior::Ignore => {}
                    WebSearchBehavior::Reject => {
                        return Err(ConversionError::bad_request(
                            "Server tools are not supported",
                        ));
                    }
                },
                ResponsesTool::Unsupported => {}
            }
        }
        Ok(converted)
    }

    pub(super) fn function_call(
        &self,
        call: ResponsesFunctionCall,
        historical_names: &mut HashMap<(String, String), String>,
    ) -> ToolCall {
        let name = match call.namespace {
            Some(namespace) => {
                let key = (namespace, call.name);
                if let Some(name) = self.namespace_names.get(&key) {
                    name.clone()
                } else if let Some(name) = historical_names.get(&key) {
                    name.clone()
                } else {
                    // Each historical namespace/name pair uses one unique name.
                    let name = unique_name(join_namespace(&key.0, &key.1), |candidate| {
                        self.is_name_taken(candidate)
                            || historical_names.values().any(|name| name == candidate)
                    });
                    historical_names.insert(key, name.clone());
                    name
                }
            }
            None => call.name,
        };
        ToolCall {
            id: call.call_id,
            name,
            arguments: call.arguments,
        }
    }

    pub(super) fn select(
        self,
        choice: Option<ResponsesToolChoice>,
        thinking: bool,
        options: &ConversionOptions,
    ) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
        // Dropping the web_search tool also drops a choice naming it.
        let choice = match choice {
            Some(ResponsesToolChoice::Named(ResponsesNamedToolChoice::WebSearch)) => {
                match options.responses_web_search {
                    WebSearchBehavior::Ignore => None,
                    WebSearchBehavior::Reject => {
                        return Err(ConversionError::bad_request(
                            "Server tools are not supported",
                        ));
                    }
                }
            }
            other => other,
        };
        // Disabling tools skips parameter-schema and final-name checks.
        // Definition conversion has already handled tool names and kinds.
        if matches!(
            choice,
            Some(ResponsesToolChoice::Mode(ResponsesToolChoiceMode::None))
        ) {
            return Ok((Vec::new(), ToolChoice::None));
        }
        let mut tools = self.definitions;
        if tools.is_empty() {
            return Ok((Vec::new(), ToolChoice::Auto));
        }
        let mut names = HashSet::new();
        for tool in &tools {
            if !names.insert(&tool.name) {
                return Err(ConversionError::bad_request("Tool names must be unique"));
            }
            if let Some(parameters) = &tool.parameters {
                validate_tool_parameters(parameters, &format!("Tool '{}' parameters", tool.name))?;
            }
        }
        let required = match choice {
            Some(ResponsesToolChoice::Named(
                ResponsesNamedToolChoice::Function { name }
                | ResponsesNamedToolChoice::Custom { name },
            )) => {
                if !names.contains(&name) {
                    return Err(ConversionError::bad_request(format!(
                        "tool_choice: no tool named '{name}' was specified"
                    )));
                }
                tools.retain(|tool| tool.name == name);
                true
            }
            Some(ResponsesToolChoice::Mode(ResponsesToolChoiceMode::Required)) => true,
            _ => false,
        };
        if required && thinking {
            return Err(ConversionError::bad_request(
                "Thinking mode does not support this tool_choice",
            ));
        }
        Ok((
            tools.into_iter().map(Into::into).collect(),
            if required {
                ToolChoice::Required
            } else {
                ToolChoice::Auto
            },
        ))
    }

    fn is_name_taken(&self, candidate: &str) -> bool {
        self.top_level_names.contains(candidate)
            || self.namespace_names.values().any(|name| name == candidate)
    }
}

impl From<ResponsesFunctionTool> for ToolDefinition {
    fn from(function: ResponsesFunctionTool) -> Self {
        Self {
            name: function.name,
            description: function.description,
            parameters: function.parameters.unwrap_or_else(|| serde_json::json!({})),
            strict: function.strict,
        }
    }
}

fn join_namespace(namespace: &str, name: &str) -> String {
    format!("{namespace}::{name}")
}

fn unique_name(base: String, is_taken: impl Fn(&str) -> bool) -> String {
    if !is_taken(&base) {
        return base;
    }
    for index in 1.. {
        let candidate = format!("{base}_{index}");
        if !is_taken(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

fn apply_patch_tool() -> ResponsesFunctionTool {
    ResponsesFunctionTool {
        name: "apply_patch".to_string(),
        description: Some(APPLY_PATCH_DESCRIPTION.to_string()),
        parameters: Some(serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The entire contents of the apply_patch command (the *** Begin Patch ... *** End Patch envelope)."
                }
            },
            "required": ["input"]
        })),
        strict: None,
    }
}

const APPLY_PATCH_DESCRIPTION: &str = "Use the `apply_patch` tool to edit files.\nYour patch language is a stripped‑down, file‑oriented diff format designed to be easy \
   to parse and safe to apply. You can think of it as a high‑level envelope:\n\n*** Begin Patch\n[ one or more file sections ]\n*** \
   End Patch\n\nWithin that envelope, you get a sequence of file operations.\nYou MUST include a header to specify the action you \
   are taking.\nEach operation starts with one of three headers:\n\n*** Add File: <path> - create a new file. Every following line \
   is a + line (the initial contents).\n*** Delete File: <path> - remove an existing file. Nothing follows.\n*** Update File: <path> \
   - patch an existing file in place (optionally with a rename).\n\nMay be immediately followed by *** Move to: <new path> if you \
   want to rename the file.\nThen one or more “hunks”, each introduced by @@ (optionally followed by a hunk header).\nWithin a hunk \
   each line starts with:\n\nFor instructions on [context_before] and [context_after]:\n- By default, show 3 lines of code \
   immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate \
   the first change’s [context_after] lines in the second change’s [context_before] lines.\n- If 3 lines of context is insufficient \
   to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the \
   snippet belongs. For instance, we might have:\n@@ class BaseClass\n[3 lines of pre-context]\n- [old_code]\n+ [new_code]\n[3 lines \
   of post-context]\n\n- If a code block is repeated so many times in a class or function such that even a single `@@` statement and \
   3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right \
   context. For instance:\n\n@@ class BaseClass\n@@ \t def method():\n[3 lines of pre-context]\n- [old_code]\n+ [new_code]\n[3 lines \
   of post-context]\n\nThe full grammar definition is below:\nPatch := Begin { FileOp } End\nBegin := \"*** Begin Patch\" \
   NEWLINE\nEnd := \"*** End Patch\" NEWLINE\nFileOp := AddFile | DeleteFile | UpdateFile\nAddFile := \"*** Add File: \" path \
   NEWLINE { \"+\" line NEWLINE }\nDeleteFile := \"*** Delete File: \" path NEWLINE\nUpdateFile := \"*** Update File: \" path \
   NEWLINE [ MoveTo ] { Hunk }\nMoveTo := \"*** Move to: \" newPath NEWLINE\nHunk := \"@@\" [ header ] NEWLINE { HunkLine } [ \"*** \
   End of File\" NEWLINE ]\nHunkLine := (\" \" | \"-\" | \"+\") text NEWLINE\n\nA full patch can combine several operations:\n\n*** \
   Begin Patch\n*** Add File: hello.txt\n+Hello world\n*** Update File: src/app.py\n*** Move to: src/main.py\n@@ def \
   greet():\n-print(\"Hi\")\n+print(\"Hello, world!\")\n*** Delete File: obsolete.txt\n*** End Patch\n\nIt is important to \
   remember:\n\n- You must include a header with your intended action (Add/Delete/Update)\n- You must prefix new lines with `+` even \
   when creating a new file\n- File references can only be relative, NEVER ABSOLUTE.\n";
