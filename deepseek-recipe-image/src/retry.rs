//! Retrying an operation that may fail.

use std::future::Future;
use std::time::Duration;

/// Delays applied before repeated attempts of an operation.
///
/// A policy with `n` delays performs at most `n + 1` attempts.
#[derive(Debug, Clone, Default)]
pub struct RetryPolicy {
    delays: Vec<Duration>,
}

impl RetryPolicy {
    /// Return a policy that waits `delays` between attempts.
    pub fn new(delays: Vec<Duration>) -> Self {
        Self { delays }
    }

    /// Return a policy that performs one attempt.
    pub const fn none() -> Self {
        Self { delays: Vec::new() }
    }

    /// Return the policy used for image downloads: one retry after 0.3 seconds,
    /// one after 0.6 seconds, and one after 1.0 second.
    pub fn download() -> Self {
        Self {
            delays: vec![
                Duration::from_millis(300),
                Duration::from_millis(600),
                Duration::from_secs(1),
            ],
        }
    }

    /// Return the maximum number of attempts.
    pub fn max_attempts(&self) -> usize {
        self.delays.len() + 1
    }

    /// Run `operation` until it succeeds, fails without retrying, or runs out of
    /// attempts.
    ///
    /// `retryable` decides whether a failure starts another attempt. `attempt`
    /// passed to `operation` is the zero-based attempt index.
    ///
    /// # Errors
    ///
    /// Returns the error of the final attempt.
    pub async fn execute<T, E, F, Fut>(
        &self,
        mut operation: F,
        retryable: impl Fn(&E) -> bool,
    ) -> Result<T, E>
    where
        F: FnMut(usize) -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        let mut attempt = 0;
        loop {
            let error = match operation(attempt).await {
                Ok(value) => return Ok(value),
                Err(error) => error,
            };
            let Some(delay) = self.delays.get(attempt) else {
                return Err(error);
            };
            if !retryable(&error) {
                return Err(error);
            }
            tokio::time::sleep(*delay).await;
            attempt += 1;
        }
    }
}
