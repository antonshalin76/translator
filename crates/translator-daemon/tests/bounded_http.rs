use std::{io, net::SocketAddr, time::Duration};

use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
    time::{Instant, sleep, timeout},
};
use translator_daemon::{ApiLimits, ControlToken, RuntimeStore, build_router, serve_control};

const TOKEN: &str = "4242424242424242424242424242424242424242424242424242424242424242";

struct Server {
    address: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<io::Result<()>>,
}

impl Server {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = build_router(
            RuntimeStore::default(),
            ControlToken::parse(TOKEN).unwrap(),
            ApiLimits::default(),
        );
        let (shutdown, receiver) = oneshot::channel();
        let task = tokio::spawn(serve_control(listener, router, async {
            let _ = receiver.await;
        }));
        Self {
            address,
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn connect(&self) -> TcpStream {
        timeout(Duration::from_secs(2), TcpStream::connect(self.address))
            .await
            .unwrap()
            .unwrap()
    }

    async fn stop(mut self) {
        let _ = self.shutdown.take().unwrap().send(());
        timeout(Duration::from_secs(7), &mut self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.abort();
    }
}

async fn status(stream: &mut TcpStream) {
    let request = format!(
        "GET /v1/status HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let response = timeout(Duration::from_secs(2), async {
        let mut response = Vec::new();
        let mut byte = [0];
        while !response.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut byte).await.unwrap();
            response.push(byte[0]);
            assert!(response.len() <= 8192);
        }
        let headers = String::from_utf8(response).unwrap();
        assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
        let length: usize = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .and_then(|value| value.parse().ok())
            })
            .unwrap();
        let mut body = vec![0; length];
        stream.read_exact(&mut body).await.unwrap();
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()
    })
    .await
    .unwrap();
    assert_eq!(response["translation_running"], false);
}

async fn closed(stream: &mut TcpStream, duration: Duration) -> bool {
    let mut byte = [0];
    matches!(
        timeout(duration, stream.read(&mut byte)).await,
        Ok(Ok(0)) | Ok(Err(_))
    )
}

#[tokio::test]
async fn sixty_fifth_connection_is_rejected_and_slots_are_reusable() {
    let server = Server::start().await;
    let mut clients = Vec::new();
    for _ in 0..64 {
        let mut stream = server.connect().await;
        status(&mut stream).await;
        clients.push(stream);
    }
    let mut excess = server.connect().await;
    let rejected = closed(&mut excess, Duration::from_millis(250)).await;
    drop(excess);
    drop(clients);
    sleep(Duration::from_millis(30)).await;
    let mut recovered = server.connect().await;
    status(&mut recovered).await;
    drop(recovered);
    server.stop().await;
    assert!(
        rejected,
        "connection 65 must close before request work is scheduled"
    );
}

#[tokio::test]
async fn slow_drip_cannot_extend_the_five_second_header_deadline() {
    let server = Server::start().await;
    let mut stream = server.connect().await;
    let start = Instant::now();
    stream.write_all(b"G").await.unwrap();
    sleep(Duration::from_millis(4800)).await;
    let early = closed(&mut stream, Duration::from_millis(10)).await;
    stream.write_all(b"E").await.unwrap();
    let expired = closed(&mut stream, Duration::from_millis(500)).await;
    let elapsed = start.elapsed();
    drop(stream);
    server.stop().await;
    assert!(!early, "header expired before five seconds");
    assert!(expired, "partial header outlived its original deadline");
    assert!(elapsed >= Duration::from_secs(5));
}

#[tokio::test]
async fn keep_alive_headers_get_a_fresh_deadline() {
    let server = Server::start().await;
    let mut stream = server.connect().await;
    sleep(Duration::from_secs(1)).await;
    status(&mut stream).await;
    let start = Instant::now();
    stream.write_all(b"G").await.unwrap();
    sleep(Duration::from_millis(4700)).await;
    let early = closed(&mut stream, Duration::from_millis(10)).await;
    let expired = closed(&mut stream, Duration::from_millis(600)).await;
    let elapsed = start.elapsed();
    drop(stream);
    server.stop().await;
    assert!(
        !early,
        "previous request's timer leaked into the next request"
    );
    assert!(expired);
    assert!(elapsed >= Duration::from_secs(5));
}

#[tokio::test]
async fn shutdown_joins_live_sse_and_partial_headers_within_its_drain_bound() {
    let server = Server::start().await;
    let mut partial = server.connect().await;
    partial.write_all(b"G").await.unwrap();
    let mut sse = server.connect().await;
    let request = format!(
        "GET /v1/events/stream HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
    );
    sse.write_all(request.as_bytes()).await.unwrap();
    let mut initial = [0; 4096];
    let read = timeout(Duration::from_secs(2), sse.read(&mut initial))
        .await
        .unwrap()
        .unwrap();
    assert!(initial[..read].starts_with(b"HTTP/1.1 200"));
    let start = Instant::now();
    server.stop().await;
    assert!(start.elapsed() < Duration::from_secs(7));
    assert!(closed(&mut partial, Duration::from_secs(1)).await);
    let mut remainder = Vec::new();
    timeout(Duration::from_secs(1), sse.read_to_end(&mut remainder))
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn owner_abort_drops_all_connection_sockets() {
    let server = Server::start().await;
    let mut stream = server.connect().await;
    status(&mut stream).await;
    server.task.abort();
    assert!(closed(&mut stream, Duration::from_secs(1)).await);
}

#[tokio::test]
async fn auth_precedes_body_parsing_over_real_tcp() {
    let server = Server::start().await;
    let mut stream = server.connect().await;
    stream.write_all(b"PATCH /v1/provider HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: 65537\r\n\r\n").await.unwrap();
    let mut response = Vec::new();
    timeout(Duration::from_secs(2), stream.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(response.starts_with(b"HTTP/1.1 401"));
    drop(stream);
    server.stop().await;
}
