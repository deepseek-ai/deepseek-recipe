//! Image fetching over HTTP with the `reqwest` client.

use std::error::Error as StdError;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use tokio_stream::StreamExt;

use crate::ImageFetcher;
use crate::error::ImageError;
use crate::limits::{ImageByteBudget, ImageLimits};

/// Number of redirects that the clients this module builds follow.
const MAX_REDIRECTS: usize = 5;

/// A rule that decides which network addresses a [`ReqwestImageFetcher`] can
/// connect to.
///
/// Install one with [`ReqwestImageFetcher::with_destination_filter`] to protect
/// a server against server-side request forgery (SSRF) when untrusted clients
/// can supply image URLs, for example by rejecting loopback, private, and
/// link-local addresses. The library does not provide a rule. A closure
/// `Fn(&str, IpAddr) -> bool` implements this trait.
pub trait DestinationFilter: Send + Sync {
    /// Return whether the fetcher can connect to `address`.
    ///
    /// `host` is the host of the image URL or of a redirect target. For a host
    /// name, `address` is one of the addresses it resolves to; for an IP
    /// address host, `host` spells `address` without IPv6 brackets.
    fn allows(&self, host: &str, address: IpAddr) -> bool;
}

impl<F> DestinationFilter for F
where
    F: Fn(&str, IpAddr) -> bool + Send + Sync,
{
    fn allows(&self, host: &str, address: IpAddr) -> bool {
        self(host, address)
    }
}

/// The destination filter rejected a host.
#[derive(Debug)]
struct RejectedDestination {
    host: String,
}

impl fmt::Display for RejectedDestination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "the destination filter rejects {}", self.host)
    }
}

impl StdError for RejectedDestination {}

/// A redirect chain is longer than [`MAX_REDIRECTS`].
#[derive(Debug)]
struct TooManyRedirects;

impl fmt::Display for TooManyRedirects {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "more than {MAX_REDIRECTS} redirects")
    }
}

impl StdError for TooManyRedirects {}

/// A DNS resolver that keeps only the addresses a destination filter allows,
/// so that the client connects to checked addresses.
struct FilteringResolver {
    filter: Arc<dyn DestinationFilter>,
}

impl reqwest::dns::Resolve for FilteringResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let filter = Arc::clone(&self.filter);
        let host = name.as_str().to_owned();
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0)).await?;
            let allowed: Vec<SocketAddr> = addresses
                .filter(|address| filter.allows(&host, address.ip()))
                .collect();
            if allowed.is_empty() {
                return Err(Box::new(RejectedDestination { host }) as _);
            }
            Ok(Box::new(allowed.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Check a URL whose host is an IP address against `filter`. A host name is
/// checked when the client resolves it.
fn validate_ip_address_host(
    url: &reqwest::Url,
    filter: &dyn DestinationFilter,
) -> Result<(), RejectedDestination> {
    let Some(host) = url.host_str() else {
        return Ok(());
    };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    match host.parse::<IpAddr>() {
        Ok(address) if !filter.allows(host, address) => Err(RejectedDestination {
            host: host.to_owned(),
        }),
        _ => Ok(()),
    }
}

/// Return the host that a destination filter rejected while `error` occurred.
fn rejected_host(error: &(dyn StdError + 'static)) -> Option<String> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(rejected) = error.downcast_ref::<RejectedDestination>() {
            return Some(rejected.host.clone());
        }
        current = error.source();
    }
    None
}

/// Return a client builder with the default timeouts.
fn default_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
}

/// An [`ImageFetcher`] that downloads images from external URLs with `reqwest`.
///
/// Clients created by [`Self::new`], [`Self::with_max_bytes`], and
/// [`Self::with_destination_filter`] follow at most five redirects and use a
/// ten-second connection timeout and a sixty-second request timeout.
/// [`Self::from_client`] preserves the supplied client's settings. Whichever
/// constructor created the fetcher, [`fetch`](ImageFetcher::fetch) limits a
/// response body to the smaller of [`Self::max_bytes`] and
/// [`ImageByteBudget::max_image_bytes`]. It rejects a response whose
/// `Content-Length` exceeds that limit before reading the body. It reads the
/// body as a stream and reserves each chunk from the budget of the resolve call
/// before keeping it, so a body that would exceed the total limit of the
/// request stops the download.
///
/// Without a [`DestinationFilter`], this fetcher connects to any address. When
/// untrusted clients can supply image URLs, the caller should install a filter
/// with [`Self::with_destination_filter`] to protect against SSRF.
pub struct ReqwestImageFetcher {
    client: reqwest::Client,
    max_bytes: usize,
    destination_filter: Option<Arc<dyn DestinationFilter>>,
}

impl ReqwestImageFetcher {
    /// Create a fetcher whose size limit is the default
    /// [`ImageLimits::max_image_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when the HTTP client cannot be constructed.
    pub fn new() -> Result<Self, ImageError> {
        Self::with_max_bytes(ImageLimits::default().max_image_bytes)
    }

    /// Create a fetcher with an explicit size limit in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError`] when the HTTP client cannot be constructed.
    pub fn with_max_bytes(max_bytes: usize) -> Result<Self, ImageError> {
        let client = default_client_builder()
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .build()
            .map_err(|err| ImageError::Client(err.to_string()))?;
        Ok(Self {
            client,
            max_bytes,
            destination_filter: None,
        })
    }

    /// Create a fetcher from an existing client with a size limit of
    /// `max_bytes` bytes. The fetcher has no destination filter.
    pub fn from_client(client: reqwest::Client, max_bytes: usize) -> Self {
        Self {
            client,
            max_bytes,
            destination_filter: None,
        }
    }

    /// Return a fetcher that connects only to the addresses `filter` allows,
    /// with the size limit of this fetcher.
    ///
    /// The filter checks the host of the image URL and of every redirect
    /// target. For a host name, the client connects only to the resolved
    /// addresses that the filter allows, so a host name that resolves to a
    /// different address later is checked again. A host with no allowed
    /// address fails the fetch with [`ImageError::DestinationRejected`].
    ///
    /// The returned fetcher has a new client with the default timeouts and
    /// redirect limit of [`Self::new`], replacing a client supplied to
    /// [`Self::from_client`]. It uses no proxy, because a proxy resolves the
    /// target host itself and the filter could not check the addresses.
    ///
    /// ```
    /// use std::net::IpAddr;
    ///
    /// use deepseek_recipe_image::ReqwestImageFetcher;
    ///
    /// let fetcher = ReqwestImageFetcher::new()?.with_destination_filter(
    ///     |_host: &str, address: IpAddr| match address.to_canonical() {
    ///         IpAddr::V4(address) => {
    ///             !(address.is_loopback()
    ///                 || address.is_private()
    ///                 || address.is_link_local())
    ///         }
    ///         IpAddr::V6(address) => {
    ///             !(address.is_loopback() || address.is_unique_local())
    ///         }
    ///     },
    /// )?;
    /// # Ok::<(), deepseek_recipe_image::ImageError>(())
    /// ```
    ///
    /// The rule in this example is not a complete SSRF policy; the caller
    /// decides which address ranges its deployment must reject.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::Client`] when the HTTP client cannot be
    /// constructed.
    pub fn with_destination_filter(
        self,
        filter: impl DestinationFilter + 'static,
    ) -> Result<Self, ImageError> {
        let filter: Arc<dyn DestinationFilter> = Arc::new(filter);
        let redirect_filter = Arc::clone(&filter);
        let client = default_client_builder()
            .no_proxy()
            .dns_resolver(Arc::new(FilteringResolver {
                filter: Arc::clone(&filter),
            }))
            .redirect(reqwest::redirect::Policy::custom(move |attempt| {
                // The first entry of `previous` is the initial URL, so this
                // check matches `reqwest::redirect::Policy::limited`.
                if attempt.previous().len() > MAX_REDIRECTS {
                    return attempt.error(TooManyRedirects);
                }
                match validate_ip_address_host(attempt.url(), redirect_filter.as_ref()) {
                    Ok(()) => attempt.follow(),
                    Err(rejected) => attempt.error(rejected),
                }
            }))
            .build()
            .map_err(|err| ImageError::Client(err.to_string()))?;
        Ok(Self {
            client,
            max_bytes: self.max_bytes,
            destination_filter: Some(filter),
        })
    }

    /// Return the encoded size limit in bytes that this fetcher applies to one
    /// image.
    ///
    /// The resolver can apply a smaller limit through the
    /// [`ImageByteBudget`] of the resolve call.
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

impl ImageFetcher for ReqwestImageFetcher {
    async fn fetch(&self, url: &str, budget: &ImageByteBudget) -> Result<Vec<u8>, ImageError> {
        // The client checks host names when it resolves them, but it connects
        // to an IP address host without resolving it.
        if let Some(filter) = &self.destination_filter
            && let Ok(parsed) = reqwest::Url::parse(url)
            && let Err(rejected) = validate_ip_address_host(&parsed, filter.as_ref())
        {
            return Err(ImageError::destination_rejected(url, rejected.host));
        }
        let response =
            self.client
                .get(url)
                .send()
                .await
                .map_err(|err| match rejected_host(&err) {
                    Some(host) => ImageError::destination_rejected(url, host),
                    None => ImageError::fetch(url, err),
                })?;
        if !response.status().is_success() {
            return Err(ImageError::fetch_status(url, response.status().as_u16()));
        }
        // The limit of the fetcher and the limit of the resolve call both apply.
        let max_bytes = self.max_bytes.min(budget.max_image_bytes());
        if let Some(length) = response.content_length()
            && length > max_bytes as u64
        {
            return Err(ImageError::image_too_large(length as usize, max_bytes));
        }

        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    // `ImageError::Fetch` is retryable, so the bytes this
                    // attempt reserved are released before the next attempt
                    // reserves the body again.
                    budget.release(body.len());
                    return Err(ImageError::fetch(url, err));
                }
            };
            let image_bytes = body.len() + chunk.len();
            if image_bytes > max_bytes {
                return Err(ImageError::image_too_large(image_bytes, max_bytes));
            }
            budget.reserve(chunk.len())?;
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    fn budget() -> ImageByteBudget {
        ImageByteBudget::new(1024, 1024, 0)
    }

    /// Serve the responses that `responses` builds from the port, in order on a
    /// loopback port, one per connection.
    async fn serve(responses: impl FnOnce(u16) -> Vec<String>) -> u16 {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("the listener binds");
        let port = listener
            .local_addr()
            .expect("the listener has an address")
            .port();
        let responses = responses(port);
        tokio::spawn(async move {
            for response in responses {
                let (mut stream, _) = listener.accept().await.expect("a client connects");
                let mut request = [0; 1024];
                let _ = stream.read(&mut request).await;
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        port
    }

    fn rejects_all() -> impl DestinationFilter {
        |_: &str, _: IpAddr| false
    }

    fn rejected(result: Result<Vec<u8>, ImageError>) -> String {
        match result {
            Err(ImageError::DestinationRejected { host, .. }) => host,
            other => panic!("expected a rejected destination, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_filter_rejects_an_ip_address_host() {
        let fetcher = ReqwestImageFetcher::new()
            .and_then(|fetcher| fetcher.with_destination_filter(rejects_all()))
            .expect("the fetcher builds");
        let result = fetcher
            .fetch("http://127.0.0.1:9/image.png", &budget())
            .await;
        assert_eq!(rejected(result), "127.0.0.1");
        let result = fetcher.fetch("http://[::1]:9/image.png", &budget()).await;
        assert_eq!(rejected(result), "::1");
    }

    #[tokio::test]
    async fn a_filter_rejects_a_host_name_with_no_allowed_address() {
        let fetcher = ReqwestImageFetcher::new()
            .and_then(|fetcher| fetcher.with_destination_filter(rejects_all()))
            .expect("the fetcher builds");
        let result = fetcher
            .fetch("http://localhost:9/image.png", &budget())
            .await;
        assert_eq!(rejected(result), "localhost");
    }

    #[tokio::test]
    async fn a_filter_rejects_a_redirect_to_a_rejected_host() {
        let port = serve(|port| {
            vec![format!(
                "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/target\r\n\
                 Content-Length: 0\r\nConnection: close\r\n\r\n"
            )]
        })
        .await;
        let fetcher = ReqwestImageFetcher::new()
            .and_then(|fetcher| {
                fetcher.with_destination_filter(|host: &str, _: IpAddr| host == "localhost")
            })
            .expect("the fetcher builds");
        let result = fetcher
            .fetch(&format!("http://localhost:{port}/start"), &budget())
            .await;
        assert_eq!(rejected(result), "127.0.0.1");
    }

    #[tokio::test]
    async fn an_allowed_destination_is_fetched() {
        let port = serve(|_| {
            vec!["HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc".to_string()]
        })
        .await;
        let fetcher = ReqwestImageFetcher::new()
            .and_then(|fetcher| fetcher.with_destination_filter(|_: &str, _: IpAddr| true))
            .expect("the fetcher builds");
        let body = fetcher
            .fetch(&format!("http://127.0.0.1:{port}/image.png"), &budget())
            .await
            .expect("the allowed destination is fetched");
        assert_eq!(body, b"abc");
    }

    /// Build `redirects` redirect responses, each to a new path on `port`,
    /// followed by a 200 response with the body `abc`.
    fn redirect_chain(port: u16, redirects: usize) -> Vec<String> {
        let mut responses: Vec<String> = (0..redirects)
            .map(|index| {
                format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/{index}\r\n\
                     Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            })
            .collect();
        responses.push(
            "HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc".to_string(),
        );
        responses
    }

    /// Return whether `result` is the failure of a redirect chain that is too
    /// long.
    fn is_too_many_redirects(result: &Result<Vec<u8>, ImageError>) -> bool {
        matches!(
            result,
            Err(ImageError::Fetch { source, .. })
                if source
                    .downcast_ref::<reqwest::Error>()
                    .is_some_and(reqwest::Error::is_redirect)
        )
    }

    #[tokio::test]
    async fn clients_follow_at_most_five_redirects() {
        let default_fetcher = ReqwestImageFetcher::new().expect("the fetcher builds");
        let filtered_fetcher = ReqwestImageFetcher::new()
            .and_then(|fetcher| fetcher.with_destination_filter(|_: &str, _: IpAddr| true))
            .expect("the fetcher builds");
        for fetcher in [default_fetcher, filtered_fetcher] {
            let port = serve(|port| redirect_chain(port, MAX_REDIRECTS)).await;
            let body = fetcher
                .fetch(&format!("http://127.0.0.1:{port}/start"), &budget())
                .await
                .expect("five redirects are followed");
            assert_eq!(body, b"abc");
            let port = serve(|port| redirect_chain(port, MAX_REDIRECTS + 1)).await;
            let result = fetcher
                .fetch(&format!("http://127.0.0.1:{port}/start"), &budget())
                .await;
            assert!(is_too_many_redirects(&result), "got {result:?}");
        }
    }
}
