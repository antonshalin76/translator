use std::{env, fs, path::PathBuf};

fn main() {
    let pipewire = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("libpipewire-0.3")
        .expect("PipeWire development headers are required");
    assert_eq!(pipewire.version, "1.0.5", "unreviewed PipeWire ABI");
    let spa_dir = pkg_config::get_variable("libspa-0.2", "plugindir")
        .expect("SPA plugin directory is required");
    let library = fs::canonicalize(PathBuf::from(spa_dir).join("aec/libspa-aec-webrtc.so"))
        .expect("installed WebRTC SPA AEC plugin is required");
    let library = library.to_str().expect("plugin path must be UTF-8");
    cc::Build::new()
        .file("src/native.c")
        .file("src/fixture.c")
        .file("src/witness.c")
        .includes(pipewire.include_paths)
        .flag("-std=c11")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-Werror")
        .define("PW_ID_INVALID", Some("SPA_ID_INVALID"))
        .define(
            "TRANSLATOR_AEC_LIBRARY",
            Some(format!("\"{library}\"").as_str()),
        )
        .compile("translator_aec_native");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=pipewire-0.3");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rerun-if-changed=src/native.c");
    println!("cargo:rerun-if-changed=src/callback_history.h");
    println!("cargo:rerun-if-changed=src/fixture.c");
    println!("cargo:rerun-if-changed=src/witness.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={library}");
    println!("cargo:warning=translator AEC plugin: {library}");
    println!("cargo:warning=translator AEC plugin identity must be pinned by the isolated runner");
    println!("cargo:rustc-env=TRANSLATOR_AEC_LIBRARY={library}");
    println!(
        "cargo:rustc-env=TRANSLATOR_PIPEWIRE_VERSION={}",
        pipewire.version
    );
    if env::var_os("CARGO_CFG_TARGET_ENDIAN").as_deref() != Some("little".as_ref()) {
        panic!("AEC IPC v1 requires a little-endian target");
    }
}
