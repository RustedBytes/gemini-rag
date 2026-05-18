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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use anyhow::Result;

    use super::collect_files;

    fn relative_paths(root: &Path, paths: Vec<std::path::PathBuf>) -> Vec<String> {
        paths
            .into_iter()
            .map(|path| {
                path.strip_prefix(root)
                    .expect("path below root")
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    #[test]
    fn collect_files_respects_recursion_hidden_files_and_size_limit() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("root");
        fs::create_dir(&root)?;
        fs::write(root.join("a.txt"), b"a")?;
        fs::write(root.join(".hidden.txt"), b"hidden")?;
        fs::write(root.join("big.txt"), b"12345")?;
        fs::create_dir(root.join("nested"))?;
        fs::write(root.join("nested").join("b.txt"), b"bb")?;

        let shallow_visible_small = collect_files(&root, false, false, Some(2))?;
        assert_eq!(relative_paths(&root, shallow_visible_small), ["a.txt"]);

        let all_files = collect_files(&root, true, true, None)?;
        assert_eq!(
            relative_paths(&root, all_files),
            [".hidden.txt", "a.txt", "big.txt", "nested/b.txt"]
        );

        Ok(())
    }

    #[test]
    fn collect_files_skips_hidden_directories_unless_requested() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("root");
        fs::create_dir(&root)?;
        fs::create_dir(root.join(".git"))?;
        fs::write(root.join(".git").join("config"), b"hidden")?;
        fs::write(root.join("visible.txt"), b"visible")?;

        let visible = collect_files(&root, true, false, None)?;
        assert_eq!(relative_paths(&root, visible), ["visible.txt"]);

        let with_hidden = collect_files(&root, true, true, None)?;
        assert_eq!(
            relative_paths(&root, with_hidden),
            [".git/config", "visible.txt"]
        );

        Ok(())
    }
}
