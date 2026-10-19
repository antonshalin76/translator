//! Provider IPC schema boundary.

mod client;
mod validation;

pub mod provider {
    tonic::include_proto!("translator.provider.v1");
}

pub use client::{
    ProviderClientError, ProviderStreamClient, authenticated_request, connect_provider,
    wait_provider_ready,
};
pub use validation::{
    MAX_ACTIVE_UTTERANCES, MAX_TERMINAL_UTTERANCES, ProviderEventValidator,
    ProviderSessionContract, ProviderValidationError, first_audio_max_age_ms,
    first_audio_target_ms, first_response_timeout_ms,
};

pub const PROVIDER_PROTO: &str =
    include_str!("../../../proto/translator/provider/v1/provider.proto");
pub const SIDECAR_SOCKET_NAME: &str = "sidecar.sock";
