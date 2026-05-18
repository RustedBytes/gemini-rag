use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn normalize_model_name(model: &str) -> String {
    match model.strip_prefix("models/").unwrap_or(model) {
        "gemini-flash-3-preview" => "gemini-3-flash-preview".to_string(),
        model => model.to_string(),
    }
}

pub(super) fn token_estimate(text: &str) -> u32 {
    text.split_whitespace()
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

pub(super) fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

pub(super) fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
