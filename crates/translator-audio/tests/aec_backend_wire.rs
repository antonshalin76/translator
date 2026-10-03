use translator_audio::{AecBackendWindow, AecBackendWire, AecMeasurementError};

const FRAME_LEN: usize = 72 + 480 * 3 * 4;

fn u32_at(body: &mut [u8], offset: usize, value: u32) {
    body[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn u64_at(body: &mut [u8], offset: usize, value: u64) {
    body[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn packet(body: Vec<u8>) -> Vec<u8> {
    let mut packet = (body.len() as u32).to_le_bytes().to_vec();
    packet.extend(body);
    packet
}

fn hello() -> Vec<u8> {
    let mut body = vec![0; 36];
    body[0] = 1;
    body[1] = 1;
    u64_at(&mut body, 4, 7);
    u64_at(&mut body, 12, 8);
    u32_at(&mut body, 20, 9);
    u32_at(&mut body, 24, 48_000);
    u32_at(&mut body, 28, 480);
    u32_at(&mut body, 32, 480);
    packet(body)
}

fn links() -> Vec<u8> {
    let mut body = vec![0; 72];
    body[0] = 6;
    body[1] = 1;
    u64_at(&mut body, 4, 7);
    u64_at(&mut body, 12, 8);
    u32_at(&mut body, 20, 9);
    for i in 0..3 {
        let at = 24 + i * 16;
        u32_at(&mut body, at, 10 + i as u32);
        u32_at(&mut body, at + 4, 20 + i as u32);
        u32_at(&mut body, at + 8, 30 + i as u32);
        u32_at(&mut body, at + 12, 40 + i as u32);
    }
    packet(body)
}

fn armed() -> Vec<u8> {
    let mut body = vec![0; 24];
    body[0] = 7;
    body[1] = 1;
    u64_at(&mut body, 4, 7);
    u64_at(&mut body, 12, 8);
    u32_at(&mut body, 20, 9);
    packet(body)
}

fn frame(sequence: u64, position: u64) -> Vec<u8> {
    let mut body = vec![0; FRAME_LEN];
    body[0] = 2;
    body[1] = 1;
    u64_at(&mut body, 4, 7);
    u64_at(&mut body, 12, 8);
    u64_at(&mut body, 20, sequence);
    u32_at(&mut body, 28, 10);
    u32_at(&mut body, 32, 480);
    u64_at(&mut body, 36, position);
    u64_at(&mut body, 44, 480);
    u32_at(&mut body, 60, 1);
    u32_at(&mut body, 64, 48_000);
    u32_at(&mut body, 68, 9);
    for i in 0..480 {
        let at = 72 + i * 12;
        body[at..at + 4].copy_from_slice(&0.125_f32.to_le_bytes());
        body[at + 4..at + 8].copy_from_slice(&0.05_f32.to_le_bytes());
        body[at + 8..at + 12].copy_from_slice(&0.025_f32.to_le_bytes());
    }
    packet(body)
}

fn decoder() -> AecBackendWire {
    let mut wire = AecBackendWire::new();
    assert!(wire.identity().is_none());
    assert!(wire.links().is_none());
    assert!(wire.push_bytes(&hello()).unwrap().is_empty());
    let identity = wire.identity().unwrap();
    assert_eq!(
        (identity.session, identity.generation, identity.node_id),
        (7, 8, 9)
    );
    assert!(wire.links().is_none());
    assert!(wire.push_bytes(&links()).unwrap().is_empty());
    let graph_links = wire.links().unwrap();
    assert_eq!(
        (
            graph_links[0].local_port,
            graph_links[0].link_id,
            graph_links[0].peer_node,
            graph_links[0].peer_port
        ),
        (10, 20, 30, 40)
    );
    assert!(wire.push_bytes(&armed()).unwrap().is_empty());
    assert_eq!(wire.accepted_frames(), 0);
    wire
}

#[test]
fn armed_is_required_once_between_links_and_first_frame() {
    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    wire.push_bytes(&links()).unwrap();
    assert_eq!(wire.finish(), Err(AecMeasurementError::Incomplete));
    assert_eq!(
        wire.push_bytes(&frame(0, 0)),
        Err(AecMeasurementError::InvalidProvenance)
    );
    assert_eq!(
        wire.push_bytes(&armed()),
        Err(AecMeasurementError::Invalidated)
    );

    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    wire.push_bytes(&links()).unwrap();
    let packet = armed();
    wire.push_bytes(&packet[..9]).unwrap();
    assert_eq!(wire.finish(), Err(AecMeasurementError::Incomplete));
    wire.push_bytes(&packet[9..]).unwrap();
    assert_eq!(wire.accepted_frames(), 0);
    wire.push_bytes(&frame(0, 0)).unwrap();
    assert_eq!(wire.accepted_frames(), 1);
    assert_eq!(
        wire.push_bytes(&armed()),
        Err(AecMeasurementError::InvalidProvenance)
    );
}

#[test]
fn armed_identity_mismatch_is_terminal() {
    for offset in [8, 16, 24] {
        let mut wire = AecBackendWire::new();
        wire.push_bytes(&hello()).unwrap();
        wire.push_bytes(&links()).unwrap();
        let mut bad = armed();
        if offset == 24 {
            u32_at(&mut bad, offset, 99);
        } else {
            u64_at(&mut bad, offset, 99);
        }
        assert_eq!(
            wire.push_bytes(&bad),
            Err(AecMeasurementError::InvalidProvenance)
        );
        assert_eq!(
            wire.push_bytes(&armed()),
            Err(AecMeasurementError::Invalidated)
        );
    }
}

#[test]
fn rejects_early_or_malformed_armed_packet() {
    let mut wire = AecBackendWire::new();
    assert_eq!(
        wire.push_bytes(&armed()),
        Err(AecMeasurementError::InvalidProvenance)
    );
    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    assert_eq!(
        wire.push_bytes(&armed()),
        Err(AecMeasurementError::InvalidProvenance)
    );
    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    wire.push_bytes(&links()).unwrap();
    let mut bad = armed();
    bad.pop();
    u32_at(&mut bad, 0, 23);
    assert_eq!(
        wire.push_bytes(&bad),
        Err(AecMeasurementError::InvalidFormat)
    );
}

#[test]
fn fragmented_wire_frames_preserve_absolute_origin_and_sequence() {
    let mut wire = decoder();
    let first = frame(100, 9_600);
    for byte in first.chunks(7) {
        assert!(wire.push_bytes(byte).unwrap().is_empty());
    }
    let mut result = Vec::new();
    for i in 1..200 {
        result.extend(wire.push_bytes(&frame(100 + i, 9_600 + i * 480)).unwrap());
    }
    assert_eq!(result.len(), 2);
    let first = result.remove(0);
    // Exhaustive destructuring makes an invented phase/acquisition label a test failure.
    let AecBackendWindow {
        identity,
        links,
        clock_id,
        xrun,
        first_sequence,
        last_sequence,
        start_sample,
        end_sample,
        raw,
        clean,
    } = first;
    assert_eq!(
        (identity.session, identity.generation, identity.node_id),
        (7, 8, 9)
    );
    assert_eq!(links[0].local_port, 10);
    assert_eq!(links[2].local_port, 12);
    assert_eq!((clock_id, xrun), (10, 0));
    assert_eq!((first_sequence, last_sequence), (100, 199));
    assert_eq!((start_sample, end_sample), (9_600, 57_600));
    assert_eq!((raw.len(), clean.len()), (48_000, 48_000));
    assert_eq!((raw[0], clean[0]), (4_096, 819));

    let next = result.pop().unwrap();
    assert_eq!((next.first_sequence, next.last_sequence), (200, 299));
    assert_eq!((next.start_sample, next.end_sample), (57_600, 105_600));
    assert_eq!((next.raw.len(), next.clean.len()), (48_000, 48_000));
    wire.finish().unwrap();
}

#[test]
fn requires_exact_hello_and_links_before_pcm() {
    let mut wire = AecBackendWire::new();
    assert_eq!(
        wire.push_bytes(&frame(0, 0)),
        Err(AecMeasurementError::InvalidProvenance)
    );

    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    assert_eq!(
        wire.push_bytes(&frame(0, 0)),
        Err(AecMeasurementError::InvalidProvenance)
    );

    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    assert_eq!(
        wire.push_bytes(&hello()),
        Err(AecMeasurementError::InvalidProvenance)
    );

    let mut wire = AecBackendWire::new();
    wire.push_bytes(&hello()).unwrap();
    let mut wrong = links();
    u64_at(&mut wrong, 8, 99);
    assert_eq!(
        wire.push_bytes(&wrong),
        Err(AecMeasurementError::InvalidProvenance)
    );
}

#[test]
fn rejects_discontinuous_or_reidentified_graph() {
    type WireFault = (&'static str, fn(&mut [u8]));
    let cases: &[WireFault] = &[
        ("session", |p| u64_at(p, 8, 99)),
        ("generation", |p| u64_at(p, 16, 99)),
        ("clock", |p| u32_at(p, 32, 99)),
        ("node", |p| u32_at(p, 72, 99)),
        ("sequence gap", |p| u64_at(p, 24, 3)),
        ("position gap", |p| u64_at(p, 40, 960)),
        ("duration", |p| u64_at(p, 48, 479)),
        ("xrun", |p| u64_at(p, 56, 1)),
        ("rate", |p| u32_at(p, 68, 44_100)),
        ("equivalent rate fraction switch", |p| {
            u32_at(p, 64, 2);
            u32_at(p, 68, 96_000);
        }),
    ];
    for (name, fault) in cases {
        let mut wire = decoder();
        wire.push_bytes(&frame(1, 0)).unwrap();
        let mut bad = frame(2, 480);
        fault(&mut bad);
        assert!(wire.push_bytes(&bad).is_err(), "{name} admitted");
        assert_eq!(
            wire.push_bytes(&frame(3, 960)),
            Err(AecMeasurementError::Invalidated)
        );
    }
}

#[test]
fn rejects_invalid_pcm_and_protocol_bounds() {
    for (name, sample) in [
        ("nan", f32::NAN),
        ("infinite", f32::INFINITY),
        ("clipped", 1.0),
    ] {
        let mut wire = decoder();
        let mut bad = frame(1, 0);
        bad[4 + 72 + 4..4 + 72 + 8].copy_from_slice(&sample.to_le_bytes());
        assert!(wire.push_bytes(&bad).is_err(), "{name} reference admitted");
    }
    let mut wire = decoder();
    assert_eq!(
        wire.push_bytes(&((4 * 1024 * 1024 + 1_u32).to_le_bytes())),
        Err(AecMeasurementError::InvalidFormat)
    );
    let mut wire = decoder();
    let mut bad = frame(1, 0);
    bad[4 + 1] = 2;
    assert_eq!(
        wire.push_bytes(&bad),
        Err(AecMeasurementError::InvalidFormat)
    );
    let mut wire = decoder();
    let partial = frame(1, 0);
    wire.push_bytes(&partial[..20]).unwrap();
    assert_eq!(wire.finish(), Err(AecMeasurementError::Incomplete));
}

#[test]
fn progress_counts_only_complete_valid_frames() {
    let mut wire = decoder();
    let first = frame(1, 0);
    wire.push_bytes(&first[..first.len() - 1]).unwrap();
    assert_eq!(wire.accepted_frames(), 0);
    wire.push_bytes(&first[first.len() - 1..]).unwrap();
    assert_eq!(wire.accepted_frames(), 1);
    let mut bad = frame(2, 480);
    bad[4 + 72..4 + 76].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(wire.push_bytes(&bad).is_err());
    assert_eq!(wire.accepted_frames(), 1);
}

#[test]
fn complete_wire_stream_is_only_generic_windows() {
    let mut wire = decoder();
    let mut windows = Vec::new();
    for i in 0..4_500_u64 {
        windows.extend(wire.push_bytes(&frame(100 + i, 9_600 + i * 480)).unwrap());
    }
    wire.finish().unwrap();
    assert_eq!(wire.accepted_frames(), 4_500);
    assert_eq!(windows.len(), 45);
    assert_eq!(
        (windows[0].start_sample, windows[0].end_sample),
        (9_600, 57_600)
    );
    assert_eq!(
        (windows[44].first_sequence, windows[44].last_sequence),
        (4_500, 4_599)
    );
    assert_eq!(
        (windows[44].start_sample, windows[44].end_sample),
        (9_600 + 44 * 48_000, 9_600 + 45 * 48_000)
    );
}

#[test]
fn fatal_is_terminal_and_cannot_be_retried_into_success() {
    let mut wire = decoder();
    let mut fatal = vec![0; 32];
    fatal[0] = 3;
    fatal[1] = 1;
    u64_at(&mut fatal, 4, 7);
    u64_at(&mut fatal, 12, 8);
    u32_at(&mut fatal, 20, 5);
    assert_eq!(
        wire.push_bytes(&packet(fatal)),
        Err(AecMeasurementError::SourceFailed)
    );
    assert_eq!(
        wire.push_bytes(&frame(0, 0)),
        Err(AecMeasurementError::Invalidated)
    );
}
