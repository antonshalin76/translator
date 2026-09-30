use std::{env, ffi::CString};

unsafe extern "C" {
    fn translator_aec_backend_run(argc: i32, argv: *const *const std::ffi::c_char) -> i32;
}

fn main() {
    let args: Vec<CString> = env::args_os()
        .map(|arg| {
            CString::new(arg.to_string_lossy().as_bytes())
                .expect("arguments must not contain NUL bytes")
        })
        .collect();
    let argv: Vec<*const std::ffi::c_char> = args.iter().map(|arg| arg.as_ptr()).collect();
    // The native boundary owns PipeWire and the WebRTC SPA plugin, not admission or proof.
    let result = unsafe { translator_aec_backend_run(argv.len() as i32, argv.as_ptr()) };
    std::process::exit(result);
}
