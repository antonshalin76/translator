use std::{
    io::{self, Read},
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use wait_timeout::ChildExt;

const SYSTEM_COMMAND_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    success: bool,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CommandResult {
    pub fn success(stdout: Vec<u8>) -> Self {
        Self {
            success: true,
            stdout,
            stderr: Vec::new(),
        }
    }

    pub fn failure(stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        Self {
            success: false,
            stdout,
            stderr,
        }
    }

    pub fn is_success(&self) -> bool {
        self.success
    }

    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    #[allow(dead_code)]
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandRunError {
    NotFound,
    SpawnFailed,
    TimedOut,
}

pub trait CommandRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        let mut child = Command::new(program)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(map_spawn_error)?;
        let stdout_reader = read_stream(child.stdout.take().ok_or(CommandRunError::SpawnFailed)?);
        let stderr_reader = read_stream(child.stderr.take().ok_or(CommandRunError::SpawnFailed)?);
        let deadline = Instant::now() + SYSTEM_COMMAND_TIMEOUT;
        let status = match child.wait_timeout(SYSTEM_COMMAND_TIMEOUT) {
            Ok(Some(status)) => status,
            Ok(None) => {
                terminate_and_reap(&mut child);
                return Err(CommandRunError::TimedOut);
            }
            Err(_) => {
                terminate_and_reap(&mut child);
                return Err(CommandRunError::SpawnFailed);
            }
        };
        let stdout = receive_stream(&stdout_reader, deadline)?;
        let stderr = receive_stream(&stderr_reader, deadline)?;
        Ok(CommandResult {
            success: status.success(),
            stdout,
            stderr,
        })
    }
}

fn map_spawn_error(error: io::Error) -> CommandRunError {
    match error.kind() {
        io::ErrorKind::NotFound => CommandRunError::NotFound,
        _ => CommandRunError::SpawnFailed,
    }
}

fn read_stream<R>(mut stream: R) -> Receiver<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut output = Vec::new();
        let result = stream.read_to_end(&mut output).map(|_| output);
        let _ = sender.send(result);
    });
    receiver
}

fn receive_stream(
    reader: &Receiver<io::Result<Vec<u8>>>,
    deadline: Instant,
) -> Result<Vec<u8>, CommandRunError> {
    let result = match reader.try_recv() {
        Ok(result) => result,
        Err(TryRecvError::Disconnected) => return Err(CommandRunError::SpawnFailed),
        Err(TryRecvError::Empty) => reader
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|_| CommandRunError::SpawnFailed)?,
    };
    result.map_err(|_| CommandRunError::SpawnFailed)
}

fn terminate_and_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
