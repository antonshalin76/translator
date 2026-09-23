use std::{sync::Arc, time::Instant};

use translator_audio::DeviceFacts;
use translator_daemon::{FactsError, RuntimeFacts, RuntimeFactsSource, RuntimeSnapshot};

pub fn fixture(snapshot: RuntimeSnapshot) -> Arc<dyn RuntimeFactsSource> {
    struct FixedFacts(RuntimeSnapshot);
    impl RuntimeFactsSource for FixedFacts {
        fn inspect(&self, deadline: Instant) -> Result<RuntimeFacts, FactsError> {
            if Instant::now() >= deadline {
                return Err(FactsError::Expired);
            }
            let devices = self.0.devices.clone().ok_or(FactsError::DiscoveryFailed)?;
            Ok(RuntimeFacts {
                devices: DeviceFacts {
                    source: devices.source,
                    sink: devices.sink,
                    output_mode: devices.acoustic.mode,
                    aec_capability: devices.acoustic.aec_capability,
                },
                audio_graph: self
                    .0
                    .audio_graph
                    .clone()
                    .ok_or(FactsError::DiscoveryFailed)?,
                routes: self.0.routes.clone().ok_or(FactsError::DiscoveryFailed)?,
            })
        }
    }
    Arc::new(FixedFacts(snapshot))
}
