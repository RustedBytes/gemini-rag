use std::{
    env,
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use env_logger::Env;
use log::Level;

static LOG_FILE: OnceLock<Mutex<File>> = OnceLock::new();

pub fn init(path: &Path) -> Result<()> {
    let env = Env::default()
        .filter_or("RUST_LOG", "info")
        .write_style_or("RUST_LOG_STYLE", "auto");
    let _ = env_logger::Builder::from_env(env)
        .format_timestamp_secs()
        .try_init();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open log file {}", path.display()))?;
    let _ = LOG_FILE.set(Mutex::new(file));
    event(format!("logging initialized: {}", path.display()));
    debug(format!(
        "environment logging configured: RUST_LOG={} RUST_LOG_STYLE={}",
        env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        env::var("RUST_LOG_STYLE").unwrap_or_else(|_| "auto".to_string())
    ));
    Ok(())
}

pub fn event(message: impl AsRef<str>) {
    write(Level::Info, message.as_ref());
}

pub fn debug(message: impl AsRef<str>) {
    write(Level::Debug, message.as_ref());
}

pub fn warn(message: impl AsRef<str>) {
    write(Level::Warn, message.as_ref());
}

pub fn error(message: impl AsRef<str>) {
    write(Level::Error, message.as_ref());
}

fn write(level: Level, message: &str) {
    log::log!(level, "{message}");

    let Some(file) = LOG_FILE.get() else {
        return;
    };
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();

    if let Ok(mut file) = file.lock() {
        let _ = writeln!(file, "{timestamp}\t{level}\t{message}");
    }
}
