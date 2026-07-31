fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure().compile_protos(
        &["../../proto/translator/provider/v1/provider.proto"],
        &["../../proto"],
    )?;
    println!("cargo:rerun-if-changed=../../proto/translator/provider/v1/provider.proto");
    Ok(())
}
