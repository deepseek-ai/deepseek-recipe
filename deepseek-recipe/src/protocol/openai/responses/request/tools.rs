use std::collections::{HashMap, HashSet};

use deepseek_recipe_core::messages::ToolCall;
use deepseek_recipe_core::tools::{ToolChoice, ToolDefinition};

use crate::protocol::client_tools::{
    ClientToolChoice, ClientToolDeclaration, convert_client_tools,
};
use crate::protocol::validation::validate_tool_name;
use crate::request::{
    ConversionError, ConversionOptions, ModelVersion, ResponseOptions, WebSearchBehavior,
};

use super::schema::{
    ResponsesFunctionCall, ResponsesFunctionTool, ResponsesInput, ResponsesInputItem,
    ResponsesNamedToolChoice, ResponsesNamespaceTool, ResponsesTool, ResponsesToolChoice,
    ResponsesToolSearchExecution, ResponsesTypedInputItem,
};

/// The namespace and local name of a namespace tool.
#[derive(PartialEq, Eq, Hash)]
pub(super) struct ResponsesNamespaceToolName {
    namespace: String,
    local_name: String,
}

/// The tool declarations of a request after name validation, with namespace
/// tools flattened into uniquely named functions.
pub(super) struct ResponsesToolSet {
    client_tools: Vec<ClientToolDeclaration>,
    /// Namespace tools in registration order: the function name and its
    /// namespace and local name. The order keeps
    /// [`Self::convert_function_call`] deterministic when duplicate namespaces
    /// register the same local name.
    namespace_tools: Vec<(String, ResponsesNamespaceToolName)>,
    top_level_names: HashSet<String>,
    model_version: ModelVersion,
    pub(super) discovered_tools: HashMap<usize, Vec<ToolDefinition>>,
    pub(super) response_options: ResponseOptions,
}

impl ResponsesToolSet {
    /// Validate tool names, flatten namespaces, and replace `apply_patch` with
    /// its function tool.
    ///
    /// # Errors
    ///
    /// Returns an error, in declaration order, for an invalid function or
    /// namespace name, a custom tool inside a namespace, an invalid or repeated
    /// function name inside a namespace, a custom tool other than
    /// `apply_patch`, or a `web_search` tool under [`WebSearchBehavior::Reject`].
    pub(super) fn from_declarations(
        tools: Option<Vec<ResponsesTool>>,
        options: &ConversionOptions,
        model_version: ModelVersion,
        input: Option<&ResponsesInput>,
    ) -> Result<Self, ConversionError> {
        let mut groups = vec![(None, tools.unwrap_or_default())];
        if model_version.supports_mid_conversation_system()
            && let Some(ResponsesInput::Items(items)) = input
        {
            for (index, item) in items.iter().enumerate() {
                if let ResponsesInputItem::Typed(ResponsesTypedInputItem::ToolSearchOutput(
                    output,
                )) = item
                {
                    groups.push((Some(index), output.tools.clone()));
                }
            }
        }
        let mut converted = Self {
            client_tools: Vec::new(),
            namespace_tools: Vec::new(),
            discovered_tools: HashMap::new(),
            top_level_names: groups
                .iter()
                .flat_map(|(_, tools)| tools)
                .filter_map(|tool| match tool {
                    ResponsesTool::Function(function) => Some(function.name.clone()),
                    ResponsesTool::Custom { name } if name == "apply_patch" => Some(name.clone()),
                    ResponsesTool::ToolSearch {
                        execution: Some(ResponsesToolSearchExecution::Client),
                        ..
                    } if model_version.supports_mid_conversation_system() => {
                        Some("tool_search".to_string())
                    }
                    _ => None,
                })
                .collect(),
            model_version,
            response_options: ResponseOptions::default(),
        };
        for (item_index, tools) in groups {
            let start = converted.client_tools.len();
            for (index, tool) in tools.into_iter().enumerate() {
                let path = item_index.map_or_else(
                    || format!("tools[{index}]"),
                    |item_index| format!("input[{item_index}].tools[{index}]"),
                );
                match tool {
                    ResponsesTool::Function(function) => {
                        validate_tool_name(&function.name, &path)?;
                        converted.client_tools.push(function.into());
                    }
                    ResponsesTool::Namespace {
                        name,
                        description,
                        tools,
                    } => {
                        validate_tool_name(&name, &path)?;
                        let mut inner_names = HashSet::new();
                        for (inner_index, tool) in tools.into_iter().enumerate() {
                            let mut function = match tool {
                                ResponsesNamespaceTool::Function(function) => function,
                                ResponsesNamespaceTool::Custom { name: custom_name } => {
                                    return Err(ConversionError::bad_request(format!(
                                        "Currently custom tools are not allowed inside a namespace. Found custom tool \
                                     '{custom_name}' in {path}.tools[{inner_index}]."
                                    )));
                                }
                            };
                            let inner_path = format!("{path}.tools[{inner_index}]");
                            validate_tool_name(&function.name, &inner_path)?;
                            if !inner_names.insert(function.name.clone()) {
                                return Err(ConversionError::bad_request(
                                    "Tool names within a namespace must be unique.",
                                ));
                            }
                            if model_version == ModelVersion::DeepseekV41Flash
                                && let Some(namespace_description) = description
                                    .as_deref()
                                    .filter(|description| !description.is_empty())
                            {
                                let function_description =
                                    function.description.as_deref().unwrap_or_default();
                                function.description = Some(format!(
                                    "{namespace_description}\n{function_description}"
                                ));
                            }
                            let function_name = unique_name(
                                join_namespace(&name, &function.name, model_version),
                                |candidate| converted.is_name_taken(candidate),
                            );
                            converted.namespace_tools.push((
                                function_name.clone(),
                                ResponsesNamespaceToolName {
                                    namespace: name.clone(),
                                    local_name: function.name,
                                },
                            ));
                            function.name = function_name;
                            converted.client_tools.push(function.into());
                        }
                    }
                    ResponsesTool::Custom { name } => {
                        if name != "apply_patch" {
                            return Err(ConversionError::bad_request(format!(
                                "Unsupported custom tool: '{name}'. Only 'apply_patch' is supported."
                            )));
                        }
                        converted.client_tools.push(apply_patch_tool(model_version));
                        converted.response_options.custom_tool_names.insert(name);
                    }
                    ResponsesTool::ToolSearch {
                        execution,
                        description,
                        parameters,
                    } => {
                        let client_tool_search =
                            execution == Some(ResponsesToolSearchExecution::Client);
                        converted.response_options.client_tool_search |=
                            item_index.is_none() && client_tool_search;
                        if model_version.supports_mid_conversation_system() && client_tool_search {
                            converted.client_tools.push(ClientToolDeclaration {
                                name: "tool_search".to_string(),
                                description,
                                parameters,
                                strict: None,
                                defer_loading: false,
                            });
                        }
                    }
                    ResponsesTool::WebSearch(_) | ResponsesTool::WebSearch20250826(_) => {
                        match options.responses_web_search {
                            WebSearchBehavior::Drop => {}
                            WebSearchBehavior::Reject => {
                                return Err(ConversionError::bad_request(
                                    "Server tools are not supported",
                                ));
                            }
                        }
                    }
                    ResponsesTool::Unsupported => {}
                }
            }
            if let Some(item_index) = item_index {
                let tools = converted.client_tools[start..]
                    .iter_mut()
                    .map(|tool| {
                        tool.defer_loading = true;
                        ToolDefinition::from(&*tool)
                    })
                    .collect();
                converted.discovered_tools.insert(item_index, tools);
            }
        }
        Ok(converted)
    }

    /// Return the function names of namespace tools mapped to their namespace
    /// and local name.
    pub(super) fn namespace_tool_map(&self) -> HashMap<String, (String, String)> {
        self.namespace_tools
            .iter()
            .map(|(function_name, name)| {
                (
                    function_name.clone(),
                    (name.namespace.clone(), name.local_name.clone()),
                )
            })
            .collect()
    }

    /// Return whether at least one tool definition was converted.
    pub(super) fn has_tools(&self) -> bool {
        !self.client_tools.is_empty()
    }

    /// Convert a historical function call into a tool call, resolving a
    /// namespaced name to its function name.
    ///
    /// A registered namespace tool uses its registered function name. Other
    /// namespaced names use one unique function name per namespace and local
    /// name pair, recorded in `historical_names`.
    pub(super) fn convert_function_call(
        &self,
        call: ResponsesFunctionCall,
        historical_names: &mut HashMap<ResponsesNamespaceToolName, String>,
    ) -> ToolCall {
        let name = match call.namespace {
            Some(namespace) => {
                let key = ResponsesNamespaceToolName {
                    namespace,
                    local_name: call.name,
                };
                let registered = self
                    .namespace_tools
                    .iter()
                    .find(|(_, name)| *name == key)
                    .map(|(function_name, _)| function_name.clone());
                if let Some(name) = registered {
                    name
                } else if let Some(name) = historical_names.get(&key) {
                    name.clone()
                } else {
                    // Each historical namespace/name pair uses one unique name.
                    let name = unique_name(
                        join_namespace(&key.namespace, &key.local_name, self.model_version),
                        |candidate| {
                            self.is_name_taken(candidate)
                                || historical_names.values().any(|name| name == candidate)
                        },
                    );
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

    /// Validate tool schemas and resolve the tool choice.
    ///
    /// A choice that names a `web_search` tool is dropped or rejected per
    /// [`ConversionOptions::responses_web_search`]. A `none` choice or an empty
    /// tool set skips the schema checks and returns no tool definitions;
    /// [`convert_client_tools`] resolves the other choices.
    ///
    /// # Errors
    ///
    /// Returns an error for a choice that names a `web_search` tool under
    /// [`WebSearchBehavior::Reject`], then the errors of
    /// [`convert_client_tools`]: duplicate function names after namespace
    /// flattening, an invalid parameter schema, a named choice that names no
    /// tool, or a named or `required` choice in thinking mode.
    pub(super) fn convert_tools(
        self,
        choice: Option<ResponsesToolChoice>,
        thinking_mode: bool,
        options: &ConversionOptions,
    ) -> Result<(Vec<ToolDefinition>, ToolChoice), ConversionError> {
        // Dropping the web_search tool also drops a choice naming it.
        let choice = match choice {
            Some(ResponsesToolChoice::Named(
                ResponsesNamedToolChoice::WebSearch | ResponsesNamedToolChoice::WebSearch20250826,
            )) => match options.responses_web_search {
                WebSearchBehavior::Drop => None,
                WebSearchBehavior::Reject => {
                    return Err(ConversionError::bad_request(
                        "Server tools are not supported",
                    ));
                }
            },
            other => other,
        };
        // Disabling tools skips parameter-schema and final-name checks.
        // Definition conversion has already handled tool names and kinds.
        if matches!(choice, Some(ResponsesToolChoice::None)) {
            return Ok((Vec::new(), ToolChoice::None));
        }
        if self.client_tools.is_empty() {
            return Ok((Vec::new(), ToolChoice::Auto));
        }
        let choice = match choice {
            Some(ResponsesToolChoice::Named(
                ResponsesNamedToolChoice::Function { name }
                | ResponsesNamedToolChoice::Custom { name },
            )) => ClientToolChoice::Named(name),
            Some(ResponsesToolChoice::Required) => ClientToolChoice::Required,
            _ => ClientToolChoice::Auto,
        };
        convert_client_tools(self.client_tools, choice, thinking_mode)
    }

    fn is_name_taken(&self, candidate: &str) -> bool {
        self.top_level_names.contains(candidate)
            || self
                .namespace_tools
                .iter()
                .any(|(function_name, _)| function_name == candidate)
    }
}

impl From<ResponsesFunctionTool> for ClientToolDeclaration {
    fn from(function: ResponsesFunctionTool) -> Self {
        Self {
            name: function.name,
            description: function.description,
            parameters: function.parameters,
            strict: function.strict,
            defer_loading: false,
        }
    }
}

/// Join a namespace and a local tool name into a function name.
fn join_namespace(namespace: &str, name: &str, model_version: ModelVersion) -> String {
    if model_version == ModelVersion::DeepseekV41Flash {
        format!("{namespace}::{name}")
    } else if namespace.ends_with('_') || name.starts_with('_') {
        format!("{namespace}{name}")
    } else {
        format!("{namespace}__{name}")
    }
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

/// Return the function tool that replaces the `apply_patch` custom tool.
fn apply_patch_tool(model_version: ModelVersion) -> ClientToolDeclaration {
    let description = match model_version {
        ModelVersion::DeepseekV4Pro => APPLY_PATCH_DESCRIPTION.replacen(
            "envelope:\n\n*** Begin Patch",
            "envelope:\n*** Begin Patch",
            1,
        ),
        ModelVersion::DeepseekV41Flash => APPLY_PATCH_DESCRIPTION.to_string(),
    };
    ClientToolDeclaration {
        name: "apply_patch".to_string(),
        description: Some(description),
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
        defer_loading: false,
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

#[cfg(test)]
mod contribution_regression {
    use super::*;

    #[test]
    fn namespaces_do_not_collide_with_synthetic_apply_patch() {
        let tools = serde_json::from_value(serde_json::json!([
            {"type":"namespace","name":"apply_","tools":[{"type":"function","name":"patch"}]},
            {"type":"custom","name":"apply_patch"}
        ]))
        .unwrap();
        let options = ConversionOptions::default();
        let set = ResponsesToolSet::from_declarations(
            Some(tools),
            &options,
            ModelVersion::DeepseekV4Pro,
            None,
        )
        .unwrap();
        let (tools, _) = set.convert_tools(None, false, &options).unwrap();
        let names: std::collections::HashSet<_> = tools.iter().map(|t| &t.name).collect();
        assert_eq!(names.len(), 2);
    }
}
