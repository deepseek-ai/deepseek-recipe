//! Configure native library linking for Python wheels.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // Wheel repair writes longer dependency paths into Mach-O load commands.
        println!("cargo::rustc-link-arg-cdylib=-Wl,-headerpad_max_install_names");
    }
}
