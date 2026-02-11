//! # rsrpc - Ergonomic Rust-to-Rust RPC
//!
//! A function-forward RPC library where the trait IS the API.
//!
//! ## Overview
//!
//! rsrpc generates RPC client and server code from a trait definition. The client
//! implements the same trait as the server, so `client.method(args)` just works.
//! No separate client types, no message enums, no schema files.
//!
//! ## Quick Start
//!
//! ```ignore
//! use anyhow::Result;
//! use rsrpc::{async_trait, Client};
//!
//! #[rsrpc::service]
//! pub trait Calculator: Send + Sync + 'static {
//!     async fn add(&self, a: i32, b: i32) -> Result<i32>;
//! }
//!
//! // Server implementation
//! struct MyCalculator;
//!
//! #[async_trait]
//! impl Calculator for MyCalculator {
//!     async fn add(&self, a: i32, b: i32) -> Result<i32> {
//!         Ok(a + b)
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Server
//!     let server = <dyn Calculator>::serve(MyCalculator);
//!     tokio::spawn(server.listen("0.0.0.0:9000"));
//!
//!     // Client
//!     let client: Client<dyn Calculator> = Client::connect("127.0.0.1:9000").await?;
//!     let result = client.add(2, 3).await?;
//!     assert_eq!(result, 5);
//!     Ok(())
//! }
//! ```
//!
//! ## Streaming
//!
//! Methods returning `Result<RpcStream<T>>` automatically stream data:
//!
//! ```ignore
//! #[rsrpc::service]
//! pub trait LogService: Send + Sync + 'static {
//!     async fn stream_logs(&self, filter: Filter) -> Result<RpcStream<LogEntry>>;
//! }
//!
//! // Client usage
//! let mut stream = client.stream_logs(filter).await?;
//! while let Some(entry) = stream.next().await {
//!     println!("{:?}", entry?);
//! }
//! ```
//!
//! ## HTTP/REST Support
//!
//! Enable the `http` feature for REST endpoint support:
//!
//! ```ignore
//! #[rsrpc::service]
//! pub trait UserService: Send + Sync + 'static {
//!     #[get("/users/{id}")]
//!     async fn get_user(&self, id: String) -> Result<User>;
//!
//!     #[post("/users")]
//!     async fn create_user(&self, user: CreateUserRequest) -> Result<User>;
//! }
//!
//! // Serve via HTTP
//! let router = <dyn UserService>::http_routes(service);
//! axum::serve(listener, router).await?;
//!
//! // Or use HTTP client
//! let client: HttpClient<dyn UserService> = HttpClient::new("http://localhost:8080");
//! client.get_user("123".into()).await?;
//! ```
//!
//! ## Features
//!
//! - `http` - Enable HTTP/REST support with axum and reqwest

mod stream;
pub use stream::*;

#[cfg(feature = "http")]
mod http_client;
#[cfg(feature = "http")]
pub use http_client::HttpClient;

use std::collections::HashMap;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use bytes::Bytes;
use serde::{de::DeserializeOwned, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{info, warn};

/// Re-export the service macro
pub use rsrpc_macro::service;

/// Re-exports for generated code
pub use async_trait::async_trait;
pub use postcard;
pub use serde;

/// Re-exports for HTTP support (only with `http` feature)
#[cfg(feature = "http")]
pub use ::http;
#[cfg(feature = "http")]
pub use axum;

// =============================================================================
// ENCODING TRAITS
// =============================================================================

/// Trait for encoding server responses into dispatch results.
///
/// This trait is automatically implemented for:
/// - `Result<T, E>` where `T: Serialize` - unary responses
/// - `Result<RpcStream<T>, E>` - streaming responses
///
/// The implementations don't conflict because `RpcStream<T>` intentionally
/// does not implement `Serialize`.
pub trait ServerEncoding {
    /// Convert this response into a dispatch result.
    fn into_dispatch(self) -> DispatchResult;
}

/// Unary response encoding for any serializable Result type.
impl<T: Serialize, E: std::fmt::Display> ServerEncoding for Result<T, E> {
    fn into_dispatch(self) -> DispatchResult {
        let wire_result: Result<T, String> = self.map_err(|e| e.to_string());
        match postcard::to_allocvec(&wire_result) {
            Ok(bytes) => DispatchResult::Unary(bytes),
            Err(e) => DispatchResult::Error(e.to_string()),
        }
    }
}

/// Streaming response encoding for RpcStream results.
/// This impl doesn't conflict with the above because RpcStream doesn't impl Serialize.
impl<T: Serialize + Unpin + Send + 'static, E: std::fmt::Display> ServerEncoding
    for Result<RpcStream<T>, E>
{
    fn into_dispatch(self) -> DispatchResult {
        match self {
            Ok(stream) => DispatchResult::Stream(Box::new(stream)),
            Err(e) => DispatchResult::Error(e.to_string()),
        }
    }
}

/// Trait for making client calls with automatic encoding/decoding.
///
/// This trait is automatically implemented for:
/// - `Result<T, anyhow::Error>` where `T: DeserializeOwned` - unary calls
/// - `Result<RpcStream<T>, anyhow::Error>` - streaming calls
pub trait ClientEncoding<Service: ?Sized + 'static>: Sized {
    /// Invoke a remote method and decode the response.
    fn invoke<R: Serialize + Sync>(
        client: &Client<Service>,
        method_id: u16,
        request: &R,
    ) -> impl Future<Output = Self> + Send;
}

/// Unary call encoding for any deserializable Result type.
impl<S: ?Sized + Sync + 'static, T: DeserializeOwned + Send> ClientEncoding<S>
    for Result<T, anyhow::Error>
{
    async fn invoke<R: Serialize + Sync>(client: &Client<S>, method_id: u16, request: &R) -> Self {
        client.call(method_id, request).await
    }
}

/// Streaming call encoding for RpcStream results.
/// This impl doesn't conflict with the above because RpcStream doesn't impl DeserializeOwned.
impl<S: ?Sized + Sync + 'static, T: DeserializeOwned + Send + 'static> ClientEncoding<S>
    for Result<RpcStream<T>, anyhow::Error>
{
    async fn invoke<R: Serialize + Sync>(client: &Client<S>, method_id: u16, request: &R) -> Self {
        client.call_stream(method_id, request).await
    }
}

// =============================================================================
// DISPATCH RESULT
// =============================================================================

/// Result from dispatching a method call.
/// Can be either a unary response or a stream of items.
pub enum DispatchResult {
    /// Unary response - single serialized payload
    Unary(Vec<u8>),
    /// Streaming response - boxed stream that yields serialized items
    Stream(Box<dyn ErasedStream + Send>),
    /// Error during dispatch
    Error(String),
}

/// Type-erased stream trait for dispatch results.
pub trait ErasedStream {
    /// Get the next item as serialized bytes.
    fn poll_next_bytes(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Vec<u8>, String>>>;
}

impl<T: Serialize + Unpin> ErasedStream for RpcStream<T> {
    fn poll_next_bytes(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Vec<u8>, String>>> {
        use futures_core::Stream;
        use std::task::Poll;

        match Stream::poll_next(self, cx) {
            Poll::Ready(Some(Ok(item))) => match postcard::to_allocvec(&item) {
                Ok(bytes) => Poll::Ready(Some(Ok(bytes))),
                Err(e) => Poll::Ready(Some(Err(e.to_string()))),
            },
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

// =============================================================================
// CLIENT
// =============================================================================

/// Handle to the background reader task.
/// When dropped, aborts the reader task to allow clean process exit.
struct ReaderHandle(tokio::task::JoinHandle<()>);

impl Drop for ReaderHandle {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A client connection to a remote RPC server.
///
/// The magic: `Client<dyn MyTrait>` implements `MyTrait`, so you can call
/// `client.method(args)` directly.
///
/// On connection loss, unary calls automatically reconnect and retry once.
pub struct Client<T: ?Sized> {
    inner: Arc<ClientInner>,
    _marker: PhantomData<T>,
}

/// Internal state for pending requests
enum PendingRequest {
    /// Unary request waiting for single response
    Unary(oneshot::Sender<Bytes>),
    /// Streaming request receiving multiple items
    Stream(mpsc::Sender<StreamFrame>),
}

/// A frame received for a streaming response
pub struct StreamFrame {
    pub frame_type: FrameType,
    pub payload: Bytes,
}

struct ClientInner {
    addr: String,
    writer: Mutex<tokio::io::WriteHalf<TcpStream>>,
    pending: Mutex<HashMap<u64, PendingRequest>>,
    next_request_id: AtomicU64,
    reader_handle: Mutex<ReaderHandle>,
}

impl<T: ?Sized> Clone for Client<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Client<T> {
    /// Connect to a remote RPC server over TCP.
    pub async fn connect(addr: &str) -> Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        let (reader, writer) = tokio::io::split(stream);

        // Use a placeholder reader handle; replaced immediately below once Arc exists.
        let inner = Arc::new(ClientInner {
            addr: addr.to_string(),
            writer: Mutex::new(writer),
            pending: Mutex::new(HashMap::new()),
            next_request_id: AtomicU64::new(1),
            reader_handle: Mutex::new(ReaderHandle(tokio::spawn(async {}))),
        });

        *inner.reader_handle.lock().await = Self::start_reader(&inner, reader);

        Ok(Self {
            inner,
            _marker: PhantomData,
        })
    }

    /// Spawn a reader task wired to our Arc<ClientInner>.
    fn start_reader(
        inner: &Arc<ClientInner>,
        reader: tokio::io::ReadHalf<TcpStream>,
    ) -> ReaderHandle {
        let inner_clone = Arc::clone(inner);
        ReaderHandle(tokio::spawn(async move {
            if let Err(e) = Self::read_responses(inner_clone, reader).await {
                // Only log if it's not a cancellation (which happens on clean shutdown)
                if !e.to_string().contains("canceled") {
                    warn!("Client reader error: {e}");
                }
            }
        }))
    }

    /// Re-establish the TCP connection to the server.
    /// Replaces the writer and reader, and clears pending requests
    /// (their oneshot senders are dropped, yielding "Request cancelled").
    async fn reconnect(&self) -> Result<()> {
        let addr = &self.inner.addr;
        let stream = TcpStream::connect(addr).await?;
        let (reader, writer) = tokio::io::split(stream);

        *self.inner.writer.lock().await = writer;
        self.inner.pending.lock().await.clear();
        // Old ReaderHandle drops here, aborting the old reader task
        *self.inner.reader_handle.lock().await = Self::start_reader(&self.inner, reader);

        info!("Reconnected to {addr}");
        Ok(())
    }

    /// Reconnect and retry up to 3 times on failure.
    async fn with_retry<R, F, Fut>(&self, f: F) -> Result<R>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<R>>,
    {
        const MAX_ATTEMPTS: usize = 3;
        let mut attempts = 0;
        loop {
            self.reconnect().await?;
            match f().await {
                Ok(val) => break Ok(val),
                Err(e) if attempts < MAX_ATTEMPTS => {
                    warn!("Call failed: {e}, retrying...");
                    attempts += 1;
                }
                Err(e) => break Err(e),
            }
        }
    }

    async fn read_responses(
        inner: Arc<ClientInner>,
        mut reader: tokio::io::ReadHalf<TcpStream>,
    ) -> Result<()> {
        loop {
            // Read header (15 bytes with frame type)
            let mut header = [0u8; STREAM_HEADER_SIZE];
            if reader.read_exact(&mut header).await.is_err() {
                break; // Connection closed
            }

            let Some((frame_type, _method_id, request_id, payload_len)) =
                decode_stream_header(&header)
            else {
                warn!("Invalid frame received");
                continue;
            };

            // Read payload
            let mut payload = vec![0u8; payload_len as usize];
            if let Err(e) = reader.read_exact(&mut payload).await {
                return Err(e.into());
            }
            let payload = Bytes::from(payload);

            // Dispatch based on request type
            let mut pending = inner.pending.lock().await;

            match frame_type {
                FrameType::Response => {
                    // Unary response - remove and complete
                    if let Some(PendingRequest::Unary(tx)) = pending.remove(&request_id) {
                        let _ = tx.send(payload);
                    }
                }
                FrameType::StreamItem => {
                    // Stream item - send to stream channel
                    if let Some(PendingRequest::Stream(tx)) = pending.get(&request_id) {
                        let _ = tx
                            .send(StreamFrame {
                                frame_type,
                                payload,
                            })
                            .await;
                    }
                }
                FrameType::StreamEnd | FrameType::StreamError => {
                    // Stream completed or errored - send final frame and remove
                    if let Some(PendingRequest::Stream(tx)) = pending.remove(&request_id) {
                        let _ = tx
                            .send(StreamFrame {
                                frame_type,
                                payload,
                            })
                            .await;
                    }
                }
                FrameType::Request => {
                    // Client shouldn't receive Request frames
                    warn!("Client received unexpected Request frame");
                }
            }
        }
        Ok(())
    }

    /// Low-level call method used by generated trait impls.
    /// Sends a request and waits for a unary response.
    /// On connection failure, reconnects and retries up to 3 times.
    pub async fn call<Req: Serialize + Sync, Resp: DeserializeOwned>(
        &self,
        method_id: u16,
        request: &Req,
    ) -> Result<Resp> {
        self.with_retry(|| self.try_call(method_id, request)).await
    }

    async fn try_call<Req: Serialize + Sync, Resp: DeserializeOwned>(
        &self,
        method_id: u16,
        request: &Req,
    ) -> Result<Resp> {
        let request_id = self.inner.next_request_id.fetch_add(1, Ordering::Relaxed);
        let payload = postcard::to_allocvec(request)?;

        // Register pending request
        let (tx, rx) = oneshot::channel();
        self.inner
            .pending
            .lock()
            .await
            .insert(request_id, PendingRequest::Unary(tx));

        // Build and send message with frame type
        let header = encode_stream_header(
            FrameType::Request,
            method_id,
            request_id,
            payload.len() as u32,
        );
        let mut message = Vec::with_capacity(STREAM_HEADER_SIZE + payload.len());
        message.extend_from_slice(&header);
        message.extend_from_slice(&payload);

        if let Err(e) = self.inner.writer.lock().await.write_all(&message).await {
            self.inner.pending.lock().await.remove(&request_id);
            return Err(e.into());
        }

        // Wait for response
        let response_payload = rx
            .await
            .map_err(|_| anyhow!("Request cancelled - connection lost"))?;
        let response: Result<Resp, String> = postcard::from_bytes(&response_payload)?;
        response.map_err(|e| anyhow!("{e}"))
    }

    /// Start a streaming call. Returns a stream of responses.
    /// On connection failure, reconnects and retries up to 3 times.
    pub async fn call_stream<Req: Serialize + Sync, Item: DeserializeOwned + Send + 'static>(
        &self,
        method_id: u16,
        request: &Req,
    ) -> Result<RpcStream<Item>> {
        self.with_retry(|| self.try_call_stream(method_id, request))
            .await
    }

    async fn try_call_stream<Req: Serialize + Sync, Item: DeserializeOwned + Send + 'static>(
        &self,
        method_id: u16,
        request: &Req,
    ) -> Result<RpcStream<Item>> {
        let request_id = self.inner.next_request_id.fetch_add(1, Ordering::Relaxed);
        let payload = postcard::to_allocvec(request)?;

        // Create channels for stream
        let (frame_tx, mut frame_rx) = mpsc::channel::<StreamFrame>(32);
        let (item_tx, item_rx) = mpsc::channel::<Result<Item, String>>(32);

        // Register pending stream
        self.inner
            .pending
            .lock()
            .await
            .insert(request_id, PendingRequest::Stream(frame_tx));

        // Spawn task to convert frames to items
        tokio::spawn(async move {
            while let Some(frame) = frame_rx.recv().await {
                match frame.frame_type {
                    FrameType::StreamItem => match postcard::from_bytes::<Item>(&frame.payload) {
                        Ok(item) => {
                            if item_tx.send(Ok(item)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = item_tx.send(Err(e.to_string())).await;
                            break;
                        }
                    },
                    FrameType::StreamEnd => {
                        break;
                    }
                    FrameType::StreamError => {
                        let error: String = postcard::from_bytes(&frame.payload)
                            .unwrap_or_else(|_| "Unknown stream error".to_string());
                        let _ = item_tx.send(Err(error)).await;
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Send request
        let header = encode_stream_header(
            FrameType::Request,
            method_id,
            request_id,
            payload.len() as u32,
        );
        let mut message = Vec::with_capacity(STREAM_HEADER_SIZE + payload.len());
        message.extend_from_slice(&header);
        message.extend_from_slice(&payload);

        if let Err(e) = self.inner.writer.lock().await.write_all(&message).await {
            self.inner.pending.lock().await.remove(&request_id);
            return Err(e.into());
        }

        Ok(RpcStream::new(item_rx))
    }
}

// =============================================================================
// SERVER
// =============================================================================

/// Type alias for dispatch handler functions.
/// Takes (service, method_id, payload) and returns either unary or streaming result.
pub type DispatchFn<T> =
    for<'a> fn(&'a T, u16, &'a [u8]) -> Pin<Box<dyn Future<Output = DispatchResult> + Send + 'a>>;

/// A server that hosts an RPC service implementation.
///
/// Use `<dyn MyTrait>::serve(impl)` to create a server.
pub struct Server<T: ?Sized> {
    service: Arc<T>,
    dispatch: DispatchFn<T>,
}

impl<T: ?Sized + Send + Sync + 'static> Server<T> {
    /// Create a server from an Arc'd service and dispatch function.
    /// Typically you should use `<dyn MyTrait>::serve(impl)` instead.
    pub fn from_arc(service: Arc<T>, dispatch: DispatchFn<T>) -> Self {
        Self { service, dispatch }
    }

    /// Listen for incoming connections on the given address.
    pub async fn listen(self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        info!("Server listening on {addr}");

        loop {
            let (stream, _peer) = listener.accept().await?;
            let service = Arc::clone(&self.service);
            let dispatch = self.dispatch;

            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, service, dispatch).await {
                    warn!("Connection error: {e}");
                }
            });
        }
    }

    async fn handle_connection(
        stream: TcpStream,
        service: Arc<T>,
        dispatch: DispatchFn<T>,
    ) -> Result<()> {
        let peer = stream
            .peer_addr()
            .map(|p| p.to_string())
            .unwrap_or_else(|_| "unknown".into());
        let (mut reader, writer) = tokio::io::split(stream);
        let writer = Arc::new(Mutex::new(writer));

        loop {
            // Read header (15 bytes with frame type)
            let mut header = [0u8; STREAM_HEADER_SIZE];
            if reader.read_exact(&mut header).await.is_err() {
                break; // Connection closed
            }

            let Some((frame_type, method_id, request_id, payload_len)) =
                decode_stream_header(&header)
            else {
                warn!("Invalid frame received from {peer}");
                continue;
            };

            // Read payload
            let mut payload = vec![0u8; payload_len as usize];
            if let Err(e) = reader.read_exact(&mut payload).await {
                return Err(e.into());
            }

            match frame_type {
                FrameType::Request => {
                    let service = Arc::clone(&service);
                    let writer = Arc::clone(&writer);

                    // Spawn each request so long-running handlers don't block
                    // subsequent requests (e.g. health-check pings) on the same connection.
                    tokio::spawn(async move {
                        let dispatch_result = dispatch(&service, method_id, &payload).await;

                        match dispatch_result {
                            DispatchResult::Unary(response_payload) => {
                                // Send unary response
                                let response_header = encode_stream_header(
                                    FrameType::Response,
                                    method_id,
                                    request_id,
                                    response_payload.len() as u32,
                                );

                                let mut response =
                                    Vec::with_capacity(STREAM_HEADER_SIZE + response_payload.len());
                                response.extend_from_slice(&response_header);
                                response.extend_from_slice(&response_payload);

                                if let Err(e) = writer.lock().await.write_all(&response).await {
                                    warn!("Failed to write response for request {request_id}: {e}");
                                }
                            }
                            DispatchResult::Stream(mut stream) => {
                                use std::future::poll_fn;
                                use std::pin::Pin;

                                loop {
                                    let item = poll_fn(|cx| {
                                        // SAFETY: The stream is boxed and we never move it
                                        let pinned = unsafe { Pin::new_unchecked(&mut *stream) };
                                        pinned.poll_next_bytes(cx)
                                    })
                                    .await;

                                    match item {
                                        Some(Ok(item_bytes)) => {
                                            let header = encode_stream_header(
                                                FrameType::StreamItem,
                                                method_id,
                                                request_id,
                                                item_bytes.len() as u32,
                                            );

                                            let mut message = Vec::with_capacity(
                                                STREAM_HEADER_SIZE + item_bytes.len(),
                                            );
                                            message.extend_from_slice(&header);
                                            message.extend_from_slice(&item_bytes);

                                            if let Err(e) =
                                                writer.lock().await.write_all(&message).await
                                            {
                                                warn!("Server stream write failed for request {request_id}: {e}");
                                                break;
                                            }
                                        }
                                        Some(Err(e)) => {
                                            // Send error and end stream
                                            let error_bytes =
                                                postcard::to_allocvec(&e).unwrap_or_default();
                                            let header = encode_stream_header(
                                                FrameType::StreamError,
                                                method_id,
                                                request_id,
                                                error_bytes.len() as u32,
                                            );

                                            let mut message = Vec::with_capacity(
                                                STREAM_HEADER_SIZE + error_bytes.len(),
                                            );
                                            message.extend_from_slice(&header);
                                            message.extend_from_slice(&error_bytes);

                                            let _ = writer.lock().await.write_all(&message).await;
                                            break;
                                        }
                                        None => {
                                            // Stream ended - send StreamEnd
                                            let header = encode_stream_header(
                                                FrameType::StreamEnd,
                                                method_id,
                                                request_id,
                                                0,
                                            );

                                            let _ = writer.lock().await.write_all(&header).await;
                                            break;
                                        }
                                    }
                                }
                            }
                            DispatchResult::Error(e) => {
                                // Send error as unary response
                                let Ok(response_payload) =
                                    postcard::to_allocvec(&Err::<(), _>(e.to_string()))
                                else {
                                    return;
                                };
                                let response_header = encode_stream_header(
                                    FrameType::Response,
                                    method_id,
                                    request_id,
                                    response_payload.len() as u32,
                                );

                                let mut response =
                                    Vec::with_capacity(STREAM_HEADER_SIZE + response_payload.len());
                                response.extend_from_slice(&response_header);
                                response.extend_from_slice(&response_payload);

                                if let Err(e) = writer.lock().await.write_all(&response).await {
                                    warn!("Failed to write error response for request {request_id}: {e}");
                                }
                            }
                        }
                    });
                }
                FrameType::StreamItem | FrameType::StreamEnd | FrameType::StreamError => {
                    // Client-side streaming frames - not yet supported
                }
                FrameType::Response => {
                    // Server shouldn't receive Response frames
                    warn!("Server received unexpected Response frame");
                }
            }
        }

        Ok(())
    }
}
