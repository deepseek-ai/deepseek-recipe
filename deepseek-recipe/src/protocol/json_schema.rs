//! Typed JSON Schema for strict tool parameters and `json_schema` output.
//!
//! Strict schemas are deserialized into this type and normalized: every object
//! lists all of its properties as required and forbids additional properties.
//! Non-strict schemas are only validated; see [`validate_schema`].

use deepseek_recipe_macros::UntaggedDeserialize;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::request::ConversionError;

/// A JSON Schema node: a single type, a `type` array, an `anyOf` list, or a
/// reference. The variant is selected by the fields present, and a failure
/// names the field that did not deserialize.
#[derive(Debug, Clone, UntaggedDeserialize, Serialize)]
#[serde(untagged)]
#[untagged_deserialize(error = "one of `type`, `anyOf`, `$ref` field is required")]
pub(crate) enum JsonSchema {
    #[untagged_deserialize(field = "type", pattern = Array(_))]
    SimpleAnyOf {
        r#type: Vec<SimpleJsonSchemaType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[untagged_deserialize(field = "type")]
    Type(JsonSchemaType),
    #[serde(rename_all = "camelCase")]
    #[untagged_deserialize(field = "anyOf")]
    AnyOf {
        any_of: Vec<JsonSchemaType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[untagged_deserialize(field = "$ref")]
    Ref {
        #[serde(rename = "$ref")]
        r#ref: String,
    },
}

/// A JSON Schema type name that a `type` array can list: `string`, `number`,
/// `integer`, `boolean`, or `null`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum SimpleJsonSchemaType {
    String,
    Number,
    Integer,
    Boolean,
    Null,
}

/// A JSON Schema node with one `type` and the keywords retained for that type.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub(crate) enum JsonSchemaType {
    #[serde(rename_all = "camelCase")]
    Object {
        properties: Option<IndexMap<String, Box<JsonSchema>>>,
        additional_properties: Option<bool>,
        required: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        // Retain legacy `$def` as well as the standard definition containers.
        #[serde(rename = "$defs", skip_serializing_if = "Option::is_none")]
        defs: Option<Box<IndexMap<String, Box<JsonSchema>>>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        definitions: Option<Box<IndexMap<String, Box<JsonSchema>>>>,
        #[serde(rename = "$def", skip_serializing_if = "Option::is_none")]
        def: Option<IndexMap<String, Box<JsonSchema>>>,
    },
    #[serde(rename_all = "camelCase")]
    String {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pattern: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<StringFormat>,
        #[serde(skip_serializing_if = "Option::is_none")]
        r#enum: Option<Vec<String>>,
    },
    #[serde(rename_all = "camelCase")]
    Number {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        r#const: Option<serde_json::Number>,
        #[serde(skip_serializing_if = "Option::is_none")]
        default: Option<serde_json::Number>,
        #[serde(skip_serializing_if = "Option::is_none")]
        minimum: Option<serde_json::Number>,
        #[serde(skip_serializing_if = "Option::is_none")]
        maximum: Option<serde_json::Number>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exclusive_minimum: Option<serde_json::Number>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exclusive_maximum: Option<serde_json::Number>,
        #[serde(skip_serializing_if = "Option::is_none")]
        multiple_of: Option<serde_json::Number>,
    },
    #[serde(rename_all = "camelCase")]
    Integer {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        r#const: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        default: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        minimum: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        maximum: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exclusive_minimum: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exclusive_maximum: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        multiple_of: Option<i64>,
    },
    #[serde(rename_all = "camelCase")]
    Boolean {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Array {
        items: Box<JsonSchema>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Null {
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

/// A `format` keyword value retained for a string schema.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum StringFormat {
    Email,
    Hostname,
    Ipv4,
    Ipv6,
    Uuid,
}

impl JsonSchema {
    /// Normalize every object in the schema for strict decoding.
    ///
    /// Each object's `required` becomes the list of all of its properties and
    /// `additionalProperties` becomes `false`. An object with no properties is
    /// rejected unless `allow_empty_object` is set, which applies to the top
    /// level only. A `required` entry that names an unknown or duplicate
    /// property, a `required` list that omits a property, and
    /// `additionalProperties: true` are rejected.
    pub(crate) fn normalize(&mut self, allow_empty_object: bool) -> Result<(), ConversionError> {
        match self {
            JsonSchema::Type(r#type) => r#type.normalize(allow_empty_object),
            JsonSchema::AnyOf { any_of, .. } => any_of
                .iter_mut()
                .try_for_each(|item| item.normalize(allow_empty_object)),
            JsonSchema::SimpleAnyOf { .. } | JsonSchema::Ref { .. } => Ok(()),
        }
    }

    /// Return whether the top level is an object.
    pub(crate) fn is_object(&self) -> bool {
        matches!(self, JsonSchema::Type(JsonSchemaType::Object { .. }))
    }
}

impl Default for JsonSchema {
    /// Return an empty object schema that accepts no properties.
    fn default() -> Self {
        JsonSchema::Type(JsonSchemaType::Object {
            properties: Some(IndexMap::new()),
            additional_properties: Some(false),
            required: Some(vec![]),
            description: None,
            def: None,
            defs: None,
            definitions: None,
        })
    }
}

impl JsonSchemaType {
    fn normalize(&mut self, allow_empty_object: bool) -> Result<(), ConversionError> {
        match self {
            JsonSchemaType::Object {
                properties: raw_properties,
                required,
                additional_properties,
                def,
                defs,
                definitions,
                ..
            } => {
                let mut properties = raw_properties.take().unwrap_or_default();
                if !allow_empty_object && properties.is_empty() {
                    return Err(ConversionError::bad_request(
                        "An object with no properties is not allowed.",
                    ));
                }
                for value in properties.values_mut() {
                    value.normalize(false)?;
                }
                for definitions in def
                    .iter_mut()
                    .chain(defs.iter_mut().map(Box::as_mut))
                    .chain(definitions.iter_mut().map(Box::as_mut))
                {
                    for value in definitions.values_mut() {
                        value.normalize(false)?;
                    }
                }

                if let Some(required) = required {
                    let mut required_set = std::collections::HashSet::new();
                    for key in required.iter() {
                        if !properties.contains_key(key) {
                            return Err(ConversionError::bad_request(format!(
                                "Required property '{key}' not found in object"
                            )));
                        }
                        if !required_set.insert(key) {
                            return Err(ConversionError::bad_request(format!(
                                "Duplicate required property '{key}'"
                            )));
                        }
                    }
                    if required_set.len() != properties.len() {
                        return Err(ConversionError::bad_request(
                            "Required properties must match all properties in the object",
                        ));
                    }
                }
                if additional_properties.is_some_and(|x| x) {
                    return Err(ConversionError::bad_request(
                        "additionalProperties should not be true",
                    ));
                }

                *required = Some(properties.keys().cloned().collect());
                *additional_properties = Some(false);
                *raw_properties = Some(properties);
            }
            JsonSchemaType::Array { items, .. } => {
                items.normalize(false)?;
            }
            JsonSchemaType::String { .. }
            | JsonSchemaType::Number { .. }
            | JsonSchemaType::Integer { .. }
            | JsonSchemaType::Boolean { .. }
            | JsonSchemaType::Null { .. } => {}
        }
        Ok(())
    }
}

/// Validate `schema` as a JSON Schema with remote references disabled.
///
/// # Errors
///
/// Returns the message of the `jsonschema` crate for an invalid schema.
pub(crate) fn validate_schema(schema: &serde_json::Value) -> Result<(), String> {
    jsonschema::options()
        .offline()
        .build(schema)
        .map(|_| ())
        .map_err(|err| err.to_string())
}

/// Normalize a strict schema value.
///
/// Returns the normalized schema. `deserialize_error` formats the message of a
/// schema that does not fit [`JsonSchema`], and `top_level_error` is the error
/// for a top level that is not an object.
pub(crate) fn normalize_strict_schema(
    schema: serde_json::Value,
    allow_empty_object: bool,
    deserialize_error: impl FnOnce(serde_json::Error) -> String,
    top_level_error: &str,
) -> Result<serde_json::Value, ConversionError> {
    let mut schema: JsonSchema = serde_json::from_value(schema)
        .map_err(|err| ConversionError::bad_request(deserialize_error(err)))?;
    schema.normalize(allow_empty_object)?;
    if !schema.is_object() {
        return Err(ConversionError::bad_request(top_level_error));
    }
    serde_json::to_value(schema)
        .map_err(|err| ConversionError::internal(format!("failed to serialize schema: {err}")))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn normalize(schema: serde_json::Value) -> Result<serde_json::Value, String> {
        normalize_strict_schema(schema, true, |err| err.to_string(), "top level")
            .map_err(|err| err.to_string())
    }

    #[test]
    fn normalize_fills_required_and_additional_properties() {
        let normalized = normalize(json!({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }))
        .expect("the schema normalizes");
        assert_eq!(normalized["required"], json!(["b", "a"]));
        assert_eq!(normalized["additionalProperties"], json!(false));
        assert_eq!(normalized["properties"]["a"]["required"], json!(["x"]));
        assert_eq!(
            normalized["properties"]["a"]["additionalProperties"],
            json!(false)
        );
    }

    #[test]
    fn normalize_rejects_inconsistent_required_lists() {
        let base = json!({"type": "object", "properties": {"a": {"type": "string"}}});
        let with = |required: serde_json::Value| {
            let mut schema = base.clone();
            schema["required"] = required;
            normalize(schema).expect_err("the schema is rejected")
        };
        assert_eq!(
            with(json!(["z"])),
            "Required property 'z' not found in object"
        );
        assert_eq!(with(json!(["a", "a"])), "Duplicate required property 'a'");
        assert_eq!(
            normalize(json!({
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
            }))
            .expect_err("the schema is rejected"),
            "Required properties must match all properties in the object"
        );
        let mut schema = base.clone();
        schema["additionalProperties"] = json!(true);
        assert_eq!(
            normalize(schema).expect_err("the schema is rejected"),
            "additionalProperties should not be true"
        );
    }

    #[test]
    fn normalize_rejects_empty_nested_objects_and_non_object_top_levels() {
        assert_eq!(
            normalize(json!({"type": "object", "properties": {"a": {"type": "object"}}}))
                .expect_err("the schema is rejected"),
            "An object with no properties is not allowed."
        );
        assert_eq!(
            normalize(json!({"type": "string"})).expect_err("the schema is rejected"),
            "top level"
        );
        assert!(
            normalize(json!({"description": "no type"}))
                .expect_err("the schema is rejected")
                .contains("one of `type`, `anyOf`, `$ref` field is required")
        );
    }
}

#[cfg(test)]
mod contribution_regression {
    use super::*;

    #[test]
    fn standard_definitions_are_retained_and_normalized() {
        for keyword in ["$defs", "definitions"] {
            let mut input = serde_json::json!({"type":"object", "properties":{"value":{"$ref":format!("#/{keyword}/Entry")}}});
            input[keyword] = serde_json::json!({"Entry":{"type":"object", "properties":{"name":{"type":"string"}}}});
            let result = normalize_strict_schema(input, true, |e| e.to_string(), "root").unwrap();
            assert_eq!(
                result[keyword]["Entry"]["required"],
                serde_json::json!(["name"])
            );
            assert_eq!(result[keyword]["Entry"]["additionalProperties"], false);
            assert!(validate_schema(&result).is_ok(), "{result}");
        }
    }
}
