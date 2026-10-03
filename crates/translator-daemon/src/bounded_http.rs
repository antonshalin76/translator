use std::{future::Future, io, time::Duration};

use axum::Router;
use hyper::server::conn::http1;
use hyper_util::{
    rt::{TokioIo, TokioTimer},
    service::TowerToHyperService,
};
use tokio::{net::TcpListener, sync::watch, task::JoinSet};

const MAX_CONNECTIONS: usize = 64;
const HEADER_TIMEOUT: Duration = Duration::from_secs(5);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

pub async fn serve_control(
    listener: TcpListener,
    router: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> io::Result<()> {
    serve_control_inner(
        listener,
        router,
        shutdown,
        #[cfg(test)]
        |_| {},
    )
    .await
}

async fn serve_control_inner(
    listener: TcpListener,
    router: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
    #[cfg(test)] observed_slots: impl Fn(usize) + Send + 'static,
) -> io::Result<()> {
    let mut connections = JoinSet::new();
    let (stop, _) = watch::channel(false);
    tokio::pin!(shutdown);
    let result = loop {
        while connections.try_join_next().is_some() {}
        #[cfg(test)]
        observed_slots(connections.len());
        tokio::select! {
            biased;
            _ = &mut shutdown => break Ok(()),
            _ = connections.join_next(), if !connections.is_empty() => {},
            accepted = listener.accept() => {
                let (socket, _) = match accepted {
                    Ok(accepted) => accepted,
                    Err(error) => break Err(error),
                };
                if connections.len() >= MAX_CONNECTIONS {
                    drop(socket);
                    continue;
                }
                let router = router.clone();
                let mut stopping = stop.subscribe();
                connections.spawn(async move {
                    let mut builder = http1::Builder::new();
                    builder.timer(TokioTimer::new()).header_read_timeout(HEADER_TIMEOUT);
                    let connection = builder.serve_connection(
                        TokioIo::new(socket), TowerToHyperService::new(router),
                    );
                    tokio::pin!(connection);
                    tokio::select! {
                        _ = &mut connection => {},
                        _ = stopping.changed() => {
                            connection.as_mut().graceful_shutdown();
                            let _ = connection.await;
                        }
                    }
                });
            }
        }
    };
    drop(listener);
    stop.send_replace(true);
    if tokio::time::timeout(DRAIN_TIMEOUT, async {
        while connections.join_next().await.is_some() {}
    })
    .await
    .is_err()
    {
        connections.shutdown().await;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpStream,
        sync::oneshot,
    };

    #[tokio::test]
    async fn connection_owner_reaps_completed_slots_and_never_schedules_excess() {
        const TEST: &str = "bounded_http::tests::connection_owner_reaps_completed_slots_and_never_schedules_excess";
        const CHILD: &str = "TRANSLATOR_HTTP_FD_CHILD";
        if std::env::var(CHILD).ok().as_deref() != Some(TEST) {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([TEST, "--exact", "--test-threads=1"])
                .env(CHILD, TEST)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(90);
            while child.try_wait().unwrap().is_none() && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let timed_out = child.try_wait().unwrap().is_none();
            if timed_out {
                child.kill().unwrap();
            }
            let output = child.wait_with_output().unwrap();
            assert!(
                !timed_out && output.status.success(),
                "isolated HTTP FD oracle failed"
            );
            let stdout = String::from_utf8(output.stdout).unwrap();
            assert!(stdout.contains(&format!("test {TEST} ... ok")));
            assert!(stdout.contains("1 passed; 0 failed; 0 ignored; 0 measured"));
            return;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let fd_count = || std::fs::read_dir("/proc/self/fd").unwrap().count();
        let baseline_fds = fd_count();
        let current = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let observed_current = current.clone();
        let observed_maximum = maximum.clone();
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(serve_control_inner(
            listener,
            Router::new(),
            async {
                let _ = stopped.await;
            },
            move |count| {
                observed_current.store(count, Ordering::SeqCst);
                observed_maximum.fetch_max(count, Ordering::SeqCst);
            },
        ));
        for _ in 0..16 {
            let mut clients = Vec::new();
            for _ in 0..MAX_CONNECTIONS {
                let mut stream = TcpStream::connect(address).await.unwrap();
                stream.write_all(b"G").await.unwrap();
                clients.push(stream);
            }
            tokio::time::timeout(Duration::from_secs(2), async {
                while current.load(Ordering::SeqCst) != MAX_CONNECTIONS {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let mut rejected = TcpStream::connect(address).await.unwrap();
            let mut byte = [0];
            let result =
                tokio::time::timeout(Duration::from_secs(1), rejected.read(&mut byte)).await;
            assert!(matches!(result, Ok(Ok(0)) | Ok(Err(_))));
            assert_eq!(maximum.load(Ordering::SeqCst), MAX_CONNECTIONS);
            drop(rejected);
            drop(clients);
            tokio::time::timeout(Duration::from_secs(2), async {
                while current.load(Ordering::SeqCst) != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(fd_count(), baseline_fds);
        }
        stop.send(()).unwrap();
        task.await.unwrap().unwrap();
        assert_eq!(fd_count(), baseline_fds - 1);
    }
}
