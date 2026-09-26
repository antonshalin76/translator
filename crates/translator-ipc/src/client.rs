use std::{path::Path, time::Duration};

use hyper_util::rt::TokioIo;
use thiserror::Error;
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{
    Code, Request,
    metadata::MetadataValue,
    transport::{Channel, Endpoint},
};
use tower::service_fn;
use uuid::Uuid;

use crate::provider::{
    ProviderEvent, ProviderProbeRequest, ProviderRequest, provider_request,
    provider_transport_client::ProviderTransportClient,
};

const PROVIDER_REQUEST_CAPACITY: usize = 20;
const PROVIDER_READY_POLL_INTERVAL: Duration = Duration::from_millis(25);
const PROBE_REQUEST_SCHEMA: &str = "translator.provider.probe_request.v1";
const PROBE_RESPONSE_SCHEMA: &str = "translator.provider.probe_response.v1";

#[derive(Debug, Error)]
pub enum ProviderClientError {
    #[error("invalid provider token")]
    InvalidToken,
    #[error("invalid provider endpoint")]
    InvalidEndpoint,
    #[error("provider transport unavailable")]
    TransportUnavailable,
    #[error("provider stream must start with an open-session request")]
    InvalidOpenRequest,
    #[error("provider request channel is closed")]
    RequestChannelClosed,
    #[error("provider event stream failed")]
    EventStreamFailed,
    #[error("provider event stream rejected the protocol")]
    EventStreamProtocol,
    #[error("provider event stream exhausted bounded capacity")]
    EventStreamResourceExhausted,
    #[error("provider event stream failed internally")]
    EventStreamInternal,
    #[error("provider event stream was cancelled")]
    EventStreamCancelled,
    #[error("provider readiness probe is invalid")]
    InvalidProbeResponse,
    #[error("provider models did not become ready before the deadline")]
    ProviderReadyTimeout,
}

pub fn authenticated_request<T>(
    message: T,
    token: &str,
) -> Result<Request<T>, ProviderClientError> {
    if token.len() != 64
        || !token
            .bytes()
            .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase())
    {
        return Err(ProviderClientError::InvalidToken);
    }
    let value: MetadataValue<_> = format!("Bearer {token}")
        .parse()
        .map_err(|_| ProviderClientError::InvalidToken)?;
    let mut request = Request::new(message);
    request.metadata_mut().insert("authorization", value);
    Ok(request)
}

pub async fn connect_provider(
    socket_path: &Path,
) -> Result<ProviderTransportClient<Channel>, ProviderClientError> {
    let path = socket_path.to_owned();
    let endpoint = Endpoint::try_from("http://[::]:50051")
        .map_err(|_| ProviderClientError::InvalidEndpoint)?;
    let channel = endpoint
        .connect_with_connector(service_fn(move |_| {
            let path = path.clone();
            async move { UnixStream::connect(path).await.map(TokioIo::new) }
        }))
        .await
        .map_err(|_| ProviderClientError::TransportUnavailable)?;
    Ok(ProviderTransportClient::new(channel))
}

pub async fn wait_provider_ready(
    socket_path: &Path,
    token: &str,
    generation_id: Uuid,
    timeout: Duration,
) -> Result<(), ProviderClientError> {
    tokio::time::timeout(timeout, async {
        loop {
            let probe = async {
                let mut client = connect_provider(socket_path).await?;
                let request = authenticated_request(
                    ProviderProbeRequest {
                        schema_version: PROBE_REQUEST_SCHEMA.into(),
                    },
                    token,
                )?;
                client
                    .probe(request)
                    .await
                    .map_err(|_| ProviderClientError::TransportUnavailable)
                    .map(|response| response.into_inner())
            }
            .await;
            if let Ok(response) = probe {
                if response.schema_version != PROBE_RESPONSE_SCHEMA
                    || response.generation_id != generation_id.to_string()
                {
                    return Err(ProviderClientError::InvalidProbeResponse);
                }
                if response.provider_ready {
                    return Ok(());
                }
            }
            tokio::time::sleep(PROVIDER_READY_POLL_INTERVAL).await;
        }
    })
    .await
    .map_err(|_| ProviderClientError::ProviderReadyTimeout)?
}

pub struct ProviderStreamClient {
    sender: mpsc::Sender<ProviderRequest>,
    events: tonic::Streaming<ProviderEvent>,
}

impl ProviderStreamClient {
    pub async fn open(
        socket_path: &Path,
        token: &str,
        open_request: ProviderRequest,
    ) -> Result<Self, ProviderClientError> {
        if !matches!(
            open_request.request,
            Some(provider_request::Request::OpenSession(_))
        ) {
            return Err(ProviderClientError::InvalidOpenRequest);
        }
        let mut client = connect_provider(socket_path).await?;
        let (sender, receiver) = mpsc::channel(PROVIDER_REQUEST_CAPACITY);
        sender
            .send(open_request)
            .await
            .map_err(|_| ProviderClientError::RequestChannelClosed)?;
        let request = authenticated_request(ReceiverStream::new(receiver), token)?;
        let events = client
            .stream(request)
            .await
            .map_err(|status| event_stream_error(status.code()))?
            .into_inner();
        Ok(Self { sender, events })
    }

    pub async fn send(&self, request: ProviderRequest) -> Result<(), ProviderClientError> {
        self.sender
            .send(request)
            .await
            .map_err(|_| ProviderClientError::RequestChannelClosed)
    }

    pub async fn next_event(&mut self) -> Result<Option<ProviderEvent>, ProviderClientError> {
        self.events
            .message()
            .await
            .map_err(|status| event_stream_error(status.code()))
    }
}

fn event_stream_error(code: Code) -> ProviderClientError {
    match code {
        Code::InvalidArgument => ProviderClientError::EventStreamProtocol,
        Code::ResourceExhausted => ProviderClientError::EventStreamResourceExhausted,
        Code::Internal => ProviderClientError::EventStreamInternal,
        Code::Cancelled => ProviderClientError::EventStreamCancelled,
        Code::Unavailable => ProviderClientError::TransportUnavailable,
        _ => ProviderClientError::EventStreamFailed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{
        OpenProviderSession, ProviderProbeResponse,
        provider_transport_server::{ProviderTransport, ProviderTransportServer},
    };
    use std::{net::Shutdown, os::unix::fs::DirBuilderExt, pin::Pin};
    use tokio::sync::oneshot;
    use tokio_stream::{Stream, StreamExt};
    use tonic::{Response, Status};

    const PRIVATE_DETAIL: &str = "private-provider-diagnostic-not-for-clients";

    struct StatusTransport {
        code: Option<Code>,
        initial: bool,
    }

    #[tonic::async_trait]
    impl ProviderTransport for StatusTransport {
        type StreamStream = Pin<Box<dyn Stream<Item = Result<ProviderEvent, Status>> + Send>>;

        async fn stream(
            &self,
            _: Request<tonic::Streaming<ProviderRequest>>,
        ) -> Result<Response<Self::StreamStream>, Status> {
            if let Some(code) = self.code
                && self.initial
            {
                return Err(Status::new(code, PRIVATE_DETAIL));
            }
            let events = self
                .code
                .map(|code| {
                    vec![
                        Ok(ProviderEvent::default()),
                        Err(Status::new(code, PRIVATE_DETAIL)),
                    ]
                })
                .unwrap_or_default();
            Ok(Response::new(Box::pin(tokio_stream::iter(events))))
        }

        async fn probe(
            &self,
            _: Request<ProviderProbeRequest>,
        ) -> Result<Response<ProviderProbeResponse>, Status> {
            Err(Status::unimplemented("probe not used"))
        }
    }

    async fn observe_transport(
        code: Option<Code>,
        initial: bool,
    ) -> Result<ProviderClientError, String> {
        let directory = std::env::temp_dir().join(format!("translator-ipc-{}", Uuid::new_v4()));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .unwrap();
        let socket = directory.join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let (stop, stopped) = oneshot::channel();
        let (remote_sender, remote_receiver) = oneshot::channel();
        let mut server = tokio::spawn(async move {
            let (connection, _) = listener.accept().await.unwrap();
            let connection = connection.into_std().unwrap();
            let remote = connection.try_clone().unwrap();
            remote_sender.send(remote).unwrap();
            let connection = UnixStream::from_std(connection).unwrap();
            let incoming = tokio_stream::once(Ok::<_, std::io::Error>(connection))
                .chain(tokio_stream::pending());
            tonic::transport::Server::builder()
                .add_service(ProviderTransportServer::new(StatusTransport {
                    code,
                    initial,
                }))
                .serve_with_incoming_shutdown(incoming, async {
                    let _ = stopped.await;
                })
                .await
        });
        let outcome = tokio::time::timeout(Duration::from_secs(3), async {
            let opened = ProviderStreamClient::open(
                &socket,
                &"ab".repeat(32),
                ProviderRequest {
                    request: Some(provider_request::Request::OpenSession(
                        OpenProviderSession::default(),
                    )),
                },
            )
            .await;
            let remote = remote_receiver.await.map_err(|_| "remote socket missing")?;
            let mut client = match opened {
                Err(error) if initial => return Ok(error),
                Err(_) => return Err("RPC failed before late-status boundary"),
                Ok(_) if initial => return Err("initial RPC status missing"),
                Ok(client) => client,
            };
            if code.is_some() {
                if client
                    .next_event()
                    .await
                    .map_err(|_| "first event failed")?
                    .is_none()
                {
                    return Err("first event missing");
                }
                return client.next_event().await.err().ok_or("late status missing");
            }
            if !matches!(client.next_event().await, Ok(None)) {
                return Err("clean EOF missing");
            }
            remote
                .shutdown(Shutdown::Both)
                .map_err(|_| "remote disconnect failed")?;
            loop {
                match client.send(ProviderRequest::default()).await {
                    Err(error) => return Ok(error),
                    Ok(()) => tokio::task::yield_now().await,
                }
            }
        })
        .await;
        let _ = stop.send(());
        let drained = tokio::time::timeout(Duration::from_secs(2), &mut server).await;
        if drained.is_err() {
            server.abort();
            let _ = server.await;
        }
        std::fs::remove_file(&socket).unwrap();
        std::fs::remove_dir(&directory).unwrap();
        if !matches!(drained, Ok(Ok(Ok(())))) {
            return Err("server cleanup failed".into());
        }
        outcome
            .map_err(|_| "transport observation timed out".to_owned())?
            .map_err(str::to_owned)
    }

    #[tokio::test]
    async fn real_uds_initial_and_late_statuses_preserve_typed_private_errors() {
        for (code, expected) in [
            (
                Code::InvalidArgument,
                ProviderClientError::EventStreamProtocol,
            ),
            (
                Code::ResourceExhausted,
                ProviderClientError::EventStreamResourceExhausted,
            ),
            (Code::Internal, ProviderClientError::EventStreamInternal),
            (Code::Cancelled, ProviderClientError::EventStreamCancelled),
            (Code::Unavailable, ProviderClientError::TransportUnavailable),
            (Code::Unknown, ProviderClientError::EventStreamFailed),
        ] {
            for initial in [true, false] {
                let error = observe_transport(Some(code), initial).await.unwrap();
                assert_eq!(
                    std::mem::discriminant(&error),
                    std::mem::discriminant(&expected),
                    "{code:?}, initial={initial}"
                );
                assert!(!error.to_string().contains(PRIVATE_DETAIL));
            }
        }
    }

    #[tokio::test]
    async fn real_uds_clean_eof_and_explicit_remote_disconnect_are_distinct() {
        let error = observe_transport(None, false).await.unwrap();
        assert!(matches!(error, ProviderClientError::RequestChannelClosed));
    }

    #[test]
    fn event_stream_status_is_reduced_to_privacy_safe_categories() {
        assert!(matches!(
            event_stream_error(Code::InvalidArgument),
            ProviderClientError::EventStreamProtocol
        ));
        assert!(matches!(
            event_stream_error(Code::ResourceExhausted),
            ProviderClientError::EventStreamResourceExhausted
        ));
        assert!(matches!(
            event_stream_error(Code::Internal),
            ProviderClientError::EventStreamInternal
        ));
        assert!(matches!(
            event_stream_error(Code::Cancelled),
            ProviderClientError::EventStreamCancelled
        ));
        assert!(matches!(
            event_stream_error(Code::Unavailable),
            ProviderClientError::TransportUnavailable
        ));
        assert!(matches!(
            event_stream_error(Code::Unknown),
            ProviderClientError::EventStreamFailed
        ));
    }
}
