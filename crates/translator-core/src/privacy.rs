use std::fmt;

use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

use crate::{AudioDirection, ProviderErrorVersion};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafeErrorCode {
    ProviderUnavailable,
    ModelNotLoaded,
    UnsupportedLanguagePair,
    QueueOverflow,
    Cancelled,
    NoSpeech,
    CloudNotEnabled,
    ProviderAuthFailed,
}

impl SafeErrorCode {
    pub const fn message(self) -> &'static str {
        match self {
            Self::ProviderUnavailable => "Provider is unavailable",
            Self::ModelNotLoaded => "Required model is not loaded",
            Self::UnsupportedLanguagePair => "Language pair is not supported",
            Self::QueueOverflow => "Provider queue limit was reached",
            Self::Cancelled => "Provider operation was cancelled",
            Self::NoSpeech => "No speech was detected",
            Self::CloudNotEnabled => "Cloud provider is not enabled",
            Self::ProviderAuthFailed => "Provider authentication failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct SafeMessage(String);

impl SafeMessage {
    fn from_code(code: SafeErrorCode) -> Self {
        Self(code.message().to_owned())
    }
}

impl<'de> Deserialize<'de> for SafeMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let candidate = String::deserialize(deserializer)?;
        let is_allowed = [
            SafeErrorCode::ProviderUnavailable,
            SafeErrorCode::ModelNotLoaded,
            SafeErrorCode::UnsupportedLanguagePair,
            SafeErrorCode::QueueOverflow,
            SafeErrorCode::Cancelled,
            SafeErrorCode::NoSpeech,
            SafeErrorCode::CloudNotEnabled,
            SafeErrorCode::ProviderAuthFailed,
        ]
        .into_iter()
        .any(|code| code.message() == candidate);
        if !is_allowed {
            return Err(serde::de::Error::custom("message is not privacy-safe"));
        }
        Ok(Self(candidate))
    }
}

impl fmt::Display for SafeMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivacySafeProviderError {
    pub schema_version: ProviderErrorVersion,
    pub session_id: Uuid,
    pub direction_id: AudioDirection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<Uuid>,
    pub event_sequence: u64,
    pub code: SafeErrorCode,
    pub retryable: bool,
    pub safe_message: SafeMessage,
}

impl PrivacySafeProviderError {
    pub fn new(
        session_id: Uuid,
        direction_id: AudioDirection,
        event_sequence: u64,
        code: SafeErrorCode,
        retryable: bool,
    ) -> Self {
        Self {
            schema_version: ProviderErrorVersion::V1,
            session_id,
            direction_id,
            stream_id: None,
            event_sequence,
            code,
            retryable,
            safe_message: SafeMessage::from_code(code),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncheckedProviderError {
    schema_version: ProviderErrorVersion,
    session_id: Uuid,
    direction_id: AudioDirection,
    stream_id: Option<Uuid>,
    event_sequence: u64,
    code: SafeErrorCode,
    retryable: bool,
    safe_message: SafeMessage,
}

impl<'de> Deserialize<'de> for PrivacySafeProviderError {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let unchecked = UncheckedProviderError::deserialize(deserializer)?;
        if unchecked.safe_message.0 != unchecked.code.message() {
            return Err(serde::de::Error::custom(
                "safe message does not match error code",
            ));
        }
        Ok(Self {
            schema_version: unchecked.schema_version,
            session_id: unchecked.session_id,
            direction_id: unchecked.direction_id,
            stream_id: unchecked.stream_id,
            event_sequence: unchecked.event_sequence,
            code: unchecked.code,
            retryable: unchecked.retryable,
            safe_message: unchecked.safe_message,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PrivacySafeLogEvent {
    pub event: &'static str,
    pub code: SafeErrorCode,
    pub retryable: bool,
}

impl From<&PrivacySafeProviderError> for PrivacySafeLogEvent {
    fn from(error: &PrivacySafeProviderError) -> Self {
        Self {
            event: "provider_error",
            code: error.code,
            retryable: error.retryable,
        }
    }
}
