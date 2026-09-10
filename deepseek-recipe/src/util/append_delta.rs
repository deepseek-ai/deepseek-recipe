/// Accumulate an incremental update into an existing value.
pub trait AppendDelta<D> {
    /// Apply one update in arrival order.
    fn append(&mut self, delta: D);
}

impl<T> AppendDelta<Option<T>> for Option<T>
where
    T: AppendDelta<T>,
{
    fn append(&mut self, delta: Option<T>) {
        let Some(delta) = delta else {
            return;
        };
        if let Some(inner) = self {
            inner.append(delta);
        } else {
            *self = Some(delta);
        }
    }
}

impl<T> AppendDelta<Vec<T>> for Vec<T> {
    fn append(&mut self, delta: Vec<T>) {
        self.extend(delta);
    }
}

impl AppendDelta<String> for String {
    fn append(&mut self, delta: String) {
        self.push_str(&delta);
    }
}

impl AppendDelta<&str> for String {
    fn append(&mut self, delta: &str) {
        self.push_str(delta);
    }
}
