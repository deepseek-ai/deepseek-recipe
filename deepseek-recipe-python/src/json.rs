//! Conversion between Python objects and JSON values.

use pyo3::exceptions::{PyOverflowError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// Maximum number of nested Python containers converted into one JSON value.
const MAX_JSON_DEPTH: usize = 128;

/// Convert a Python object into a JSON value.
///
/// Accepts `None`, `bool`, `int`, `float`, `str`, `list`, `tuple`, and `dict`
/// with string keys.
///
/// # Errors
///
/// Raises `TypeError` for unsupported types, `OverflowError` for an integer
/// outside the JSON number range, and `ValueError` for a non-finite float or a
/// circular reference. More than 128 nested containers raises `RecursionError`.
pub(crate) fn python_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    python_to_json_with_parents(value, &mut Vec::new())
}

fn python_to_json_with_parents(
    value: &Bound<'_, PyAny>,
    parents: &mut Vec<*mut pyo3::ffi::PyObject>,
) -> PyResult<Value> {
    if value.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(boolean) = value.cast::<PyBool>() {
        return Ok(Value::Bool(boolean.is_true()));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Value::String(text.to_str()?.to_owned()));
    }
    if let Ok(integer) = value.cast::<PyInt>() {
        if let Ok(number) = integer.extract::<i64>() {
            return Ok(Value::Number(Number::from(number)));
        }
        if let Ok(number) = integer.extract::<u64>() {
            return Ok(Value::Number(Number::from(number)));
        }
        return Err(PyOverflowError::new_err(
            "integer is outside the JSON number range",
        ));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        let number = float.value();
        return Number::from_f64(number)
            .map(Value::Number)
            .ok_or_else(|| PyValueError::new_err(format!("{number} is not a finite JSON number")));
    }
    let pointer = value.as_ptr();
    if parents.contains(&pointer) {
        return Err(PyValueError::new_err("circular reference in JSON input"));
    }
    if parents.len() >= MAX_JSON_DEPTH {
        return Err(PyRecursionError::new_err(format!(
            "JSON input exceeds the maximum nesting depth of {MAX_JSON_DEPTH}"
        )));
    }
    parents.push(pointer);
    let result = python_container_to_json(value, parents);
    parents.pop();
    result
}

fn python_container_to_json(
    value: &Bound<'_, PyAny>,
    parents: &mut Vec<*mut pyo3::ffi::PyObject>,
) -> PyResult<Value> {
    if let Ok(list) = value.cast::<PyList>() {
        return Ok(Value::Array(
            list.iter()
                .map(|item| python_to_json_with_parents(&item, parents))
                .collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return Ok(Value::Array(
            tuple
                .iter()
                .map(|item| python_to_json_with_parents(&item, parents))
                .collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut entries = Map::new();
        for (key, item) in dict.iter() {
            let key = key
                .cast_into::<PyString>()
                .map_err(|_| PyTypeError::new_err("JSON object keys must be str"))?;
            entries.insert(
                key.to_str()?.to_owned(),
                python_to_json_with_parents(&item, parents)?,
            );
        }
        return Ok(Value::Object(entries));
    }
    Err(PyTypeError::new_err(format!(
        "{} is not a JSON value",
        value.get_type().name()?
    )))
}

/// Convert a JSON value into a Python object.
pub(crate) fn json_to_python(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(boolean) => boolean.into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                integer.into_pyobject(py)?.into_any().unbind()
            } else if let Some(integer) = number.as_u64() {
                integer.into_pyobject(py)?.into_any().unbind()
            } else {
                number
                    .as_f64()
                    .unwrap_or_default()
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        Value::String(text) => PyString::new(py, text).into_any().unbind(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_python(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, json_to_python(py, item)?)?;
            }
            dict.into_any().unbind()
        }
    })
}
