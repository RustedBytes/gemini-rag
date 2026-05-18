use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

pub fn init(path: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open log file {}", path.display()))?;
    let _ = LOG_FILE.set(Mutex::new(file));
    event(format!("logging initialized: {}", path.display()));
    Ok(())
}

pub fn event(message: impl AsRef<str>) {
    let Some(file) = LOG_FILE.get() else {
        return;
    };
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();

    if let Ok(mut file) = file.lock() {
        let _ = writeln!(file, "{timestamp}\t{}", message.as_ref());
    }
}
