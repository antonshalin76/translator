pub fn install_private_panic_hook() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        std::panic::set_hook(Box::new(|_| {
            use std::io::Write;
            let _ = std::io::stderr()
                .write_all(b"{\"event\":\"runtime_panic\",\"code\":\"internal_error\"}\n");
        }));
    });
}
