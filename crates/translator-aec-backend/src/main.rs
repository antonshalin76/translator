use std::{env, ffi::CString};

unsafe extern "C" {
    fn translator_aec_backend_run(argc: i32, argv: *const *const std::ffi::c_char) -> i32;
    fn translator_aec_physical_run(argc: i32, argv: *const *const std::ffi::c_char) -> i32;
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
    let result = unsafe {
        if args
            .get(1)
            .is_some_and(|arg| arg.as_bytes() == b"--physical")
        {
            translator_aec_physical_run(argv.len() as i32, argv.as_ptr())
        } else {
            translator_aec_backend_run(argv.len() as i32, argv.as_ptr())
        }
    };
    std::process::exit(result);
}

#[cfg(test)]
mod physical_guards {
    unsafe extern "C" {
        fn translator_aec_physical_cursor_guard(
            written: u64,
            delay: i64,
            buffer: u64,
            previous: u64,
            running: i32,
            transfer: i64,
            played: *mut u64,
        ) -> i32;
        fn translator_aec_physical_timestamp_guard(
            previous: u64,
            current: u64,
            running: i32,
        ) -> i32;
    }
    #[test]
    fn actual_consumption_is_not_queued_reference() {
        let mut played = 0;
        assert_eq!(
            unsafe {
                translator_aec_physical_cursor_guard(3840, 1920, 3840, 0, 1, 480, &mut played)
            },
            0
        );
        assert_eq!(played, 1920);
        assert_ne!(played, 3840);
        for (written, delay, buffer, previous, running, transfer) in [
            (3840, -1, 3840, 0, 1, 480),
            (3840, 3841, 3840, 0, 1, 480),
            (480, 481, 3840, 0, 1, 480),
            (3840, 1920, 3840, 1921, 1, 480),
            (3840, 1920, 3840, 0, 0, 480),
            (3840, 1920, 3840, 0, 1, 479),
            (3840, 1920, 3840, 0, 1, -32),
            (3840, 1920, 3840, 0, 1, 481),
        ] {
            assert_ne!(
                unsafe {
                    translator_aec_physical_cursor_guard(
                        written,
                        delay,
                        buffer,
                        previous,
                        running,
                        transfer,
                        &mut played,
                    )
                },
                0
            );
        }
    }
    #[test]
    fn unknown_hardware_timestamp_does_not_hide_loss_or_regression() {
        assert_eq!(
            unsafe { translator_aec_physical_timestamp_guard(0, 0, 1) },
            1
        );
        assert_eq!(
            unsafe { translator_aec_physical_timestamp_guard(50, 0, 1) },
            1
        );
        assert_eq!(
            unsafe { translator_aec_physical_timestamp_guard(50, 50, 1) },
            0
        );
        assert!(unsafe { translator_aec_physical_timestamp_guard(50, 49, 1) } < 0);
        assert!(unsafe { translator_aec_physical_timestamp_guard(0, 0, 0) } < 0);
    }
}
