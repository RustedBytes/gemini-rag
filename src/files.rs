use std::path::{Path, PathBuf};

use anyhow::Result;
use walkdir::{DirEntry, WalkDir};

use crate::logging;

pub fn collect_files(
    folder: &Path,
    recursive: bool,
    include_hidden: bool,
    max_bytes: Option<u64>,
) -> Result<Vec<PathBuf>> {
    logging::event(format!(
        "collect files: folder={} recursive={recursive} include_hidden={include_hidden} max_bytes={max_bytes:?}",
        folder.display()
    ));
    let max_depth = if recursive { usize::MAX } else { 1 };
    let walker = WalkDir::new(folder)
        .max_depth(max_depth)
        .into_iter()
        .filter_entry(|entry| include_hidden || !is_hidden(entry));

    let mut files = walker
        .filter_map(|entry| match entry {
            Ok(entry) => inspect_file_entry(entry, max_bytes).transpose(),
            Err(error) => Some(Err(error.into())),
        })
        .collect::<Result<Vec<_>>>()?;
    files.sort();
    logging::event(format!("collect files complete: count={}", files.len()));
    Ok(files)
}

fn inspect_file_entry(entry: DirEntry, max_bytes: Option<u64>) -> Result<Option<PathBuf>> {
    if !entry.file_type().is_file() {
        return Ok(None);
    }

    let metadata = entry.metadata()?;
    logging::debug(format!(
        "discovered file: path={} bytes={}",
        entry.path().display(),
        metadata.len()
    ));

    if max_bytes.is_some_and(|limit| metadata.len() > limit) {
        logging::warn(format!(
            "skip file larger than max_bytes: path={} bytes={} max_bytes={max_bytes:?}",
            entry.path().display(),
            metadata.len()
        ));
        eprintln!(
            "Skipping {} because it is larger than --max-bytes",
            entry.path().display()
        );
        return Ok(None);
    }

    Ok(Some(entry.into_path()))
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with('.'))
}
