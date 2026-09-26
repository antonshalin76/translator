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
    DeadlineExpired,
}

pub trait CommandRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError>;

    fn run(&self, program: &str, args: &[String]) -> Result<CommandResult, CommandRunError> {
        self.run_until(program, args, Instant::now() + SYSTEM_COMMAND_TIMEOUT)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    fn run_until(
        &self,
        program: &str,
        args: &[String],
        deadline: Instant,
    ) -> Result<CommandResult, CommandRunError> {
        let now = Instant::now();
        if now >= deadline {
            return Err(CommandRunError::DeadlineExpired);
        }
        let local_deadline = now + SYSTEM_COMMAND_TIMEOUT;
        let timeout_error = if deadline <= local_deadline {
            CommandRunError::DeadlineExpired
        } else {
            CommandRunError::TimedOut
        };
        let deadline = deadline.min(local_deadline);
        let mut child = Command::new(program)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(map_spawn_error)?;
        let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
            terminate_and_reap(&mut child);
            return Err(CommandRunError::SpawnFailed);
        };
        let stdout_reader = match read_stream(stdout) {
            Ok(reader) => reader,
            Err(error) => {
                terminate_and_reap(&mut child);
                return Err(error);
            }
        };
        let stderr_reader = match read_stream(stderr) {
            Ok(reader) => reader,
            Err(error) => {
                terminate_and_reap(&mut child);
                let _ = stdout_reader.1.join();
                return Err(error);
            }
        };
        let status = match child.wait_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Some(status)) => Ok(status),
            Ok(None) => {
                terminate_and_reap(&mut child);
                Err(timeout_error)
            }
            Err(_) => {
                terminate_and_reap(&mut child);
                Err(CommandRunError::SpawnFailed)
            }
        };
        let output = status.and_then(|status| {
            let stdout = receive_stream(&stdout_reader.0, deadline, timeout_error)?;
            let stderr = receive_stream(&stderr_reader.0, deadline, timeout_error)?;
            Ok((status, stdout, stderr))
        });
        let stdout_join = stdout_reader.1.join();
        let stderr_join = stderr_reader.1.join();
        if stdout_join.is_err() || stderr_join.is_err() {
            return Err(CommandRunError::SpawnFailed);
        }
        let (status, stdout, stderr) = output?;
        if Instant::now() >= deadline {
            return Err(timeout_error);
        }
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

type StreamReader = (Receiver<io::Result<Vec<u8>>>, thread::JoinHandle<()>);

fn read_stream<R>(mut stream: R) -> Result<StreamReader, CommandRunError>
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    let handle = thread::Builder::new()
        .name("tr-cmd-reader".to_owned())
        .spawn(move || {
            let mut output = Vec::new();
            let result = stream.read_to_end(&mut output).map(|_| output);
            let _ = sender.send(result);
        })
        .map_err(|_| CommandRunError::SpawnFailed)?;
    Ok((receiver, handle))
}

fn receive_stream(
    reader: &Receiver<io::Result<Vec<u8>>>,
    deadline: Instant,
    timeout_error: CommandRunError,
) -> Result<Vec<u8>, CommandRunError> {
    let result = match reader.try_recv() {
        Ok(result) => result,
        Err(TryRecvError::Disconnected) => return Err(CommandRunError::SpawnFailed),
        Err(TryRecvError::Empty) => reader
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .map_err(|error| match error {
                mpsc::RecvTimeoutError::Timeout => timeout_error,
                mpsc::RecvTimeoutError::Disconnected => CommandRunError::SpawnFailed,
            })?,
    };
    result.map_err(|_| CommandRunError::SpawnFailed)
}

fn terminate_and_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
