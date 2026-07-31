use std::{
    fs,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use tempfile::tempdir;
use tonic::Code;
use translator_daemon::{
    GRACEFUL_SHUTDOWN_TIMEOUT, ProcessSidecarRuntime, SidecarSupervisor, SupervisorError,
};
use translator_ipc::{
    ProviderEventValidator, ProviderSessionContract, ProviderStreamClient, authenticated_request,
    connect_provider,
    provider::{
        AudioDirection, Language, OpenProviderSession, PcmFormat, ProviderId, ProviderProbeRequest,
        ProviderRequest, SampleFormat, TranslationMode, VoiceEngine, VoiceGender, VoiceProfile,
        provider_request,
    },
    wait_provider_ready,
};
use uuid::Uuid;

const WRONG_TOKEN: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn session_contract(session_id: Uuid, direction: AudioDirection) -> ProviderSessionContract {
    let (source_language, target_language) = match direction {
        AudioDirection::Microphone => (Language::Ru, Language::En),
        AudioDirection::Speaker => (Language::En, Language::Ru),
        AudioDirection::Unspecified => unreachable!(),
    };
    ProviderSessionContract {
        session_id,
        stream_id: Uuid::nil(),
        provider_id: ProviderId::Local,
        direction_id: direction,
        source_language,
        target_language,
        mode: TranslationMode::QualityFirst,
        input_format: PcmFormat {
            sample_rate_hz: 16_000,
            channels: 1,
            sample_format: SampleFormat::S16le.into(),
            frame_duration_ms: 100,
        },
        output_format: PcmFormat {
            sample_rate_hz: 16_000,
            channels: 1,
            sample_format: SampleFormat::S16le.into(),
            frame_duration_ms: 100,
        },
        debug_text_enabled: false,
    }
}

fn open_request(contract: &ProviderSessionContract) -> ProviderRequest {
    let input_format = contract.input_format;
    let output_format = contract.output_format;
    ProviderRequest {
        request: Some(provider_request::Request::OpenSession(
            OpenProviderSession {
                schema_version: "translator.provider.open_session.v1".into(),
                session_id: contract.session_id.to_string(),
                provider_id: contract.provider_id.into(),
                direction_id: contract.direction_id.into(),
                source_language: contract.source_language.into(),
                target_language: contract.target_language.into(),
                mode: contract.mode.into(),
                requested_input_format: Some(input_format),
                requested_output_format: Some(output_format),
                voice_profile: Some(VoiceProfile {
                    language: contract.target_language.into(),
                    gender: VoiceGender::Male.into(),
                    engine: VoiceEngine::Piper.into(),
                    model_path: None,
                    provider_voice_id: None,
                }),
                debug_text_enabled: contract.debug_text_enabled,
            },
        )),
    }
}

async fn open_session(
    socket_path: &Path,
    token: &str,
    contract: ProviderSessionContract,
) -> ProviderStreamClient {
    let mut stream = ProviderStreamClient::open(socket_path, token, open_request(&contract))
        .await
        .unwrap();
    let mut validator = ProviderEventValidator::new(contract);

    let opened = stream.next_event().await.unwrap().unwrap();
    validator.validate(&opened, 0).unwrap();
    let health = stream.next_event().await.unwrap().unwrap();
    validator.validate(&health, 0).unwrap();

    stream
}

fn process_runtime(
    root: &Path,
    python: PathBuf,
    socket_path: PathBuf,
    expected_uid: u32,
) -> ProcessSidecarRuntime {
    let launcher = socket_path
        .parent()
        .unwrap()
        .join(format!("unavailable-runtime-{}", Uuid::new_v4()));
    fs::write(
        &launcher,
        format!(
            "#!/bin/sh\nexport TRANSLATOR_LOCAL_RUNTIME_MODE=unavailable\nexec {} \"$@\"\n",
            python.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&launcher, PermissionsExt::from_mode(0o700)).unwrap();
    ProcessSidecarRuntime::new(launcher, root.join("sidecar"), socket_path, expected_uid).unwrap()
}

async fn assert_rejects_wrong_token(socket_path: &Path) {
    let mut client = connect_provider(socket_path).await.unwrap();
    let request = authenticated_request(
        ProviderProbeRequest {
            schema_version: "translator.provider.probe_request.v1".into(),
        },
        WRONG_TOKEN,
    )
    .unwrap();
    let error = client.probe(request).await.unwrap_err();
    assert_eq!(error.code(), Code::Unauthenticated);
}

async fn assert_matching_probe(socket_path: &Path, token: &str, generation_id: Uuid) {
    let mut client = connect_provider(socket_path).await.unwrap();
    let request = authenticated_request(
        ProviderProbeRequest {
            schema_version: "translator.provider.probe_request.v1".into(),
        },
        token,
    )
    .unwrap();
    let response = client.probe(request).await.unwrap().into_inner();
    assert_eq!(
        response.schema_version,
        "translator.provider.probe_response.v1"
    );
    assert_eq!(response.generation_id, generation_id.to_string());
}

#[tokio::test]
async fn rust_supervisor_starts_authenticated_python_sidecar_and_opens_duplex_sessions() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = workspace_root();
        let python = root.join("sidecar/.venv/bin/python");
        assert!(
            python.is_file(),
            "Task 5 integration requires sidecar/.venv/bin/python"
        );
        let runtime_dir = tempdir().unwrap();
        fs::set_permissions(runtime_dir.path(), PermissionsExt::from_mode(0o700)).unwrap();
        let expected_uid = fs::metadata(runtime_dir.path()).unwrap().uid();
        let socket_path = runtime_dir.path().join("sidecar.sock");
        let runtime = process_runtime(&root, python, socket_path.clone(), expected_uid);
        let mut supervisor = SidecarSupervisor::new(runtime);

        supervisor.start().await.unwrap();
        let launch = supervisor.launch().unwrap().clone();
        let child_pid = supervisor.runtime().child_pid().unwrap();
        let command_line = fs::read(format!("/proc/{child_pid}/cmdline")).unwrap();
        assert!(
            !command_line
                .windows(launch.token.len())
                .any(|window| window == launch.token.as_bytes())
        );
        assert_matching_probe(&socket_path, &launch.token, launch.generation_id).await;
        wait_provider_ready(
            &socket_path,
            &launch.token,
            launch.generation_id,
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_rejects_wrong_token(&socket_path).await;

        let microphone_id = Uuid::new_v4();
        let speaker_id = Uuid::new_v4();
        let (microphone, speaker) = tokio::join!(
            open_session(
                &socket_path,
                &launch.token,
                session_contract(microphone_id, AudioDirection::Microphone,),
            ),
            open_session(
                &socket_path,
                &launch.token,
                session_contract(speaker_id, AudioDirection::Speaker),
            )
        );
        supervisor.register_session(microphone_id).unwrap();
        supervisor.register_session(speaker_id).unwrap();
        assert_eq!(supervisor.active_sessions().len(), 2);
        assert_eq!(supervisor.status_handle().active_session_count(), 2);

        supervisor.shutdown().await.unwrap();
        drop((microphone, speaker));
        assert_eq!(supervisor.runtime().last_reaped_pid(), Some(child_pid));
        assert!(!Path::new(&format!("/proc/{child_pid}")).exists());
        assert!(!socket_path.exists());
    })
    .await
    .expect("real sidecar lifecycle exceeded ten seconds");
}

#[tokio::test]
async fn real_python_sidecar_with_wrong_generation_is_reaped_and_rejected() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = workspace_root();
        let python = root.join("sidecar/.venv/bin/python");
        let runtime_dir = tempdir().unwrap();
        fs::set_permissions(runtime_dir.path(), PermissionsExt::from_mode(0o700)).unwrap();
        let expected_uid = fs::metadata(runtime_dir.path()).unwrap().uid();
        let socket_path = runtime_dir.path().join("sidecar.sock");
        let wrapper = runtime_dir.path().join("wrong-generation-python");
        let wrong_generation = Uuid::new_v4();
        fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\nexport TRANSLATOR_SIDECAR_GENERATION={wrong_generation}\nexec {} \"$@\"\n",
                python.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&wrapper, PermissionsExt::from_mode(0o700)).unwrap();
        let runtime = process_runtime(
            &root,
            wrapper,
            socket_path.clone(),
            expected_uid,
        );
        let mut supervisor = SidecarSupervisor::new(runtime);

        assert_eq!(
            supervisor.start().await.unwrap_err(),
            SupervisorError::GenerationMismatch
        );
        let reaped = supervisor.runtime().last_reaped_pid().unwrap();
        assert!(!Path::new(&format!("/proc/{reaped}")).exists());
        assert!(!socket_path.exists());
        assert!(!supervisor.is_ready());
    })
    .await
    .expect("wrong-generation sidecar lifecycle exceeded ten seconds");
}

#[tokio::test]
async fn normal_shutdown_delivers_sigterm_before_reap() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = workspace_root();
        let python = root.join("sidecar/.venv/bin/python");
        let runtime_dir = tempdir().unwrap();
        fs::set_permissions(runtime_dir.path(), PermissionsExt::from_mode(0o700)).unwrap();
        let expected_uid = fs::metadata(runtime_dir.path()).unwrap().uid();
        let socket_path = runtime_dir.path().join("sidecar.sock");
        let marker = runtime_dir.path().join("sigterm-observed");
        let wrapper = runtime_dir.path().join("term-observing-python");
        fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\ntrap 'touch {}; kill -TERM \"$child\"; wait \"$child\"; exit 0' TERM\n{} \"$@\" &\nchild=$!\nwait \"$child\"\n",
                marker.display(),
                python.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&wrapper, PermissionsExt::from_mode(0o700)).unwrap();
        let runtime = process_runtime(&root, wrapper, socket_path.clone(), expected_uid);
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();

        supervisor.shutdown().await.unwrap();

        assert!(marker.exists());
        assert!(!socket_path.exists());
    })
    .await
    .expect("graceful sidecar shutdown exceeded ten seconds");
}

#[tokio::test]
async fn ignored_sigterm_escalates_to_bounded_kill_and_reap() {
    assert!(GRACEFUL_SHUTDOWN_TIMEOUT <= Duration::from_secs(2));
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = workspace_root();
        let python = root.join("sidecar/.venv/bin/python");
        let runtime_dir = tempdir().unwrap();
        fs::set_permissions(runtime_dir.path(), PermissionsExt::from_mode(0o700)).unwrap();
        let expected_uid = fs::metadata(runtime_dir.path()).unwrap().uid();
        let socket_path = runtime_dir.path().join("sidecar.sock");
        let wrapper = runtime_dir.path().join("term-ignoring-python");
        let term_marker = runtime_dir.path().join("term-before-kill");
        let child_pid_file = runtime_dir.path().join("sidecar-child-pid");
        fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\ntrap 'touch {}' TERM\n{} \"$@\" &\nchild=$!\necho \"$child\" > {}\nwhile true; do wait \"$child\"; done\n",
                term_marker.display(),
                python.display(),
                child_pid_file.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&wrapper, PermissionsExt::from_mode(0o700)).unwrap();
        let runtime = process_runtime(&root, wrapper, socket_path.clone(), expected_uid);
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let pid = supervisor.runtime().child_pid().unwrap();
        let child_pid = fs::read_to_string(&child_pid_file)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();

        let shutdown_started = std::time::Instant::now();
        supervisor.shutdown().await.unwrap();
        let shutdown_elapsed = shutdown_started.elapsed();

        assert!(term_marker.exists());
        assert!(shutdown_elapsed >= GRACEFUL_SHUTDOWN_TIMEOUT);
        assert!(shutdown_elapsed <= GRACEFUL_SHUTDOWN_TIMEOUT + Duration::from_secs(1));
        assert_eq!(supervisor.runtime().last_reaped_pid(), Some(pid));
        assert!(!Path::new(&format!("/proc/{pid}")).exists());
        assert!(!Path::new(&format!("/proc/{child_pid}")).exists());
        assert!(!socket_path.exists());
    })
    .await
    .expect("forced sidecar shutdown exceeded ten seconds");
}

#[tokio::test]
async fn leader_exit_does_not_leave_a_process_group_descendant_alive() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = workspace_root();
        let python = root.join("sidecar/.venv/bin/python");
        let runtime_dir = tempdir().unwrap();
        fs::set_permissions(runtime_dir.path(), PermissionsExt::from_mode(0o700)).unwrap();
        let expected_uid = fs::metadata(runtime_dir.path()).unwrap().uid();
        let socket_path = runtime_dir.path().join("sidecar.sock");
        let wrapper = runtime_dir.path().join("early-exit-python");
        let descendant_pid_file = runtime_dir.path().join("descendant-pid");
        fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\ntrap 'exit 0' TERM\n{} \"$@\" &\nsidecar=$!\nsh -c 'trap \"\" TERM; while true; do sleep 1; done' &\necho \"$!\" > {}\nwait \"$sidecar\"\n",
                python.display(),
                descendant_pid_file.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&wrapper, PermissionsExt::from_mode(0o700)).unwrap();
        let runtime = process_runtime(&root, wrapper, socket_path.clone(), expected_uid);
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let leader_pid = supervisor.runtime().child_pid().unwrap();
        let descendant_pid = fs::read_to_string(&descendant_pid_file)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();

        let shutdown_result = supervisor.shutdown().await;
        let descendant_survived =
            Path::new(&format!("/proc/{descendant_pid}")).exists();
        if descendant_survived {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{leader_pid}")])
                .status();
            for _ in 0..100 {
                if !Path::new(&format!("/proc/{descendant_pid}")).exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        assert_eq!(shutdown_result, Ok(()));
        assert!(
            !descendant_survived,
            "shutdown returned while a process-group descendant was alive"
        );
        assert!(
            !Path::new(&format!("/proc/{descendant_pid}")).exists(),
            "test cleanup failed to reap the process-group descendant"
        );
        assert!(!socket_path.exists());
    })
    .await
    .expect("early leader exit shutdown exceeded ten seconds");
}

#[tokio::test]
async fn reaped_leader_does_not_discard_ownership_of_live_process_group() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let root = workspace_root();
        let python = root.join("sidecar/.venv/bin/python");
        let runtime_dir = tempdir().unwrap();
        fs::set_permissions(runtime_dir.path(), PermissionsExt::from_mode(0o700)).unwrap();
        let expected_uid = fs::metadata(runtime_dir.path()).unwrap().uid();
        let socket_path = runtime_dir.path().join("sidecar.sock");
        let wrapper = runtime_dir.path().join("pre-exiting-python");
        let descendant_pid_file = runtime_dir.path().join("sidecar-pid");
        let exit_gate = runtime_dir.path().join("allow-wrapper-exit");
        fs::write(
            &wrapper,
            format!(
                "#!/bin/sh\n{} \"$@\" &\necho \"$!\" > {}\nwhile [ ! -f {} ]; do sleep 0.01; done\nexit 0\n",
                python.display(),
                descendant_pid_file.display(),
                exit_gate.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&wrapper, PermissionsExt::from_mode(0o700)).unwrap();
        let runtime = process_runtime(&root, wrapper, socket_path.clone(), expected_uid);
        let mut supervisor = SidecarSupervisor::new(runtime);
        supervisor.start().await.unwrap();
        let leader_pid = supervisor.runtime().child_pid().unwrap();
        let descendant_pid = fs::read_to_string(&descendant_pid_file)
            .unwrap()
            .trim()
            .parse::<u32>()
            .unwrap();
        fs::write(&exit_gate, b"exit").unwrap();

        for _ in 0..100 {
            if !supervisor.is_ready() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!supervisor.is_ready());
        assert!(Path::new(&format!("/proc/{descendant_pid}")).exists());

        let shutdown_result = supervisor.shutdown().await;
        let descendant_survived =
            Path::new(&format!("/proc/{descendant_pid}")).exists();
        if descendant_survived {
            let _ = Command::new("kill")
                .args(["-KILL", "--", &format!("-{leader_pid}")])
                .status();
            for _ in 0..100 {
                if !Path::new(&format!("/proc/{descendant_pid}")).exists() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }

        assert_eq!(shutdown_result, Ok(()));
        assert!(
            !descendant_survived,
            "shutdown lost ownership of a live process-group descendant"
        );
        assert!(
            !Path::new(&format!("/proc/{descendant_pid}")).exists(),
            "test cleanup failed to reap the process-group descendant"
        );
        assert!(!socket_path.exists());
    })
    .await
    .expect("pre-exited leader shutdown exceeded ten seconds");
}
