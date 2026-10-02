fn main() {
    assert!(
        std::env::var("PROFILE").as_deref() != Ok("release") || !tauri_build::is_dev(),
        "release UI requires --features translator-ui/custom-protocol"
    );
    tauri_build::build();
}
