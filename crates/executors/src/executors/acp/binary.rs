//! Registry agents shipped as a per-platform archive instead of an npm package.
//!
//! The archive is downloaded once, checked against the registry's sha256 and
//! unpacked under the cache dir. A new registry version unpacks next to the old
//! one, so sessions still running the old binary are left alone.

use std::{
    collections::HashMap,
    path::{Component, Path, PathBuf},
    sync::LazyLock,
    time::Duration,
};

use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::{io::AsyncWriteExt, sync::Mutex};

use super::registry::{BinaryTarget, registry_key};
use crate::command::CommandBuilder;

const INSTALL_DIR_NAME: &str = "acp-agents";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Longest silence tolerated mid-download; archives are too big for a total timeout
const READ_TIMEOUT: Duration = Duration::from_secs(60);

/// One install at a time, so two spawns of a new agent don't both download it
static INSTALL_LOCK: Mutex<()> = Mutex::const_new(());

static HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build()
        .expect("failed to build ACP binary HTTP client")
});

#[derive(Debug, Error)]
pub enum BinaryError {
    #[error("failed to download ACP agent '{id}': {source}")]
    Download {
        id: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("failed to install ACP agent '{id}': {source}")]
    Io {
        id: String,
        #[source]
        source: std::io::Error,
    },
    #[error("ACP agent '{id}' failed its checksum: expected {expected}, got {actual}")]
    ChecksumMismatch {
        id: String,
        expected: String,
        actual: String,
    },
    #[error("ACP agent '{id}' has an archive this app can't install: {archive}")]
    UnsupportedArchive { id: String, archive: String },
    #[error("ACP agent '{id}' has an invalid command '{cmd}'")]
    InvalidCommand { id: String, cmd: String },
    #[error("ACP agent '{id}' archive doesn't contain its command '{cmd}'")]
    MissingCommand { id: String, cmd: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArchiveKind {
    TarGz,
    Zip,
}

impl ArchiveKind {
    /// `None` for anything but an https `.tar.gz`/`.tgz`/`.zip`.
    fn of(archive: &str) -> Option<Self> {
        let url = reqwest::Url::parse(archive).ok()?;
        if url.scheme() != "https" {
            return None;
        }
        let path = url.path().to_ascii_lowercase();
        if path.ends_with(".tar.gz") || path.ends_with(".tgz") {
            Some(Self::TarGz)
        } else if path.ends_with(".zip") {
            Some(Self::Zip)
        } else {
            None
        }
    }
}

/// Registry platform key of this machine, e.g. `darwin-aarch64`.
pub(crate) fn host_platform() -> Option<String> {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        "linux" => "linux",
        "windows" => "windows",
        _ => return None,
    };
    let arch = match std::env::consts::ARCH {
        arch @ ("aarch64" | "x86_64") => arch,
        _ => return None,
    };
    let libc = if os == "linux" && is_musl_host() {
        "-musl"
    } else {
        ""
    };
    Some(format!("{os}-{arch}{libc}"))
}

/// The host's libc, not this binary's: a static musl build also runs on glibc.
fn is_musl_host() -> bool {
    std::fs::read_dir("/lib").is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().starts_with("ld-musl-"))
    })
}

/// The target for this machine, if it's one this app can install.
pub fn host_target(targets: &HashMap<String, BinaryTarget>) -> Option<&BinaryTarget> {
    let target = targets.get(&host_platform()?)?;
    (ArchiveKind::of(&target.archive).is_some() && relative_cmd(&target.cmd).is_some())
        .then_some(target)
}

/// `cmd` as a path inside the unpacked archive. `None` if it's empty, absolute
/// or climbs out with `..`: the registry must not pick a program outside it.
fn relative_cmd(cmd: &str) -> Option<PathBuf> {
    let mut relative = PathBuf::new();
    for component in Path::new(&cmd.replace('\\', "/")).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => relative.push(part),
            _ => return None,
        }
    }
    (!relative.as_os_str().is_empty()).then_some(relative)
}

fn install_root() -> PathBuf {
    workspace_utils::cache_dir().join(INSTALL_DIR_NAME)
}

/// `<root>/<AGENT_KEY>/<version>-<hash>`. The hash covers the archive URL and
/// checksum, so a republished archive never reuses an older unpack.
fn install_dir(root: &Path, id: &str, version: &str, target: &BinaryTarget) -> PathBuf {
    let version: String = version
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let digest = Sha256::new()
        .chain_update(&target.archive)
        .chain_update(&target.sha256)
        .finalize();
    root.join(registry_key(id).as_str())
        .join(format!("v{version}-{:.12}", format!("{digest:x}")))
}

/// The command that runs agent `id`, downloading and unpacking it first if
/// this version isn't installed yet.
pub async fn command(
    id: &str,
    version: &str,
    target: &BinaryTarget,
) -> Result<CommandBuilder, BinaryError> {
    let program = ensure_installed(&install_root(), id, version, target).await?;
    Ok(CommandBuilder::new(quote(id, &program)?).extend_params(target.args.clone()))
}

/// Deletes every installed version of agent `id`.
pub async fn remove(id: &str) -> std::io::Result<()> {
    let dir = install_root().join(registry_key(id).as_str());
    match tokio::fs::remove_dir_all(dir).await {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// The base is split like a shell line, so a path with spaces must be quoted.
fn quote(id: &str, program: &Path) -> Result<String, BinaryError> {
    let invalid = || BinaryError::InvalidCommand {
        id: id.to_string(),
        cmd: program.display().to_string(),
    };
    let program = program.to_str().ok_or_else(invalid)?;
    #[cfg(windows)]
    {
        if program.contains('"') {
            return Err(invalid());
        }
        Ok(format!("\"{program}\""))
    }
    #[cfg(not(windows))]
    {
        shlex::try_quote(program)
            .map(|quoted| quoted.into_owned())
            .map_err(|_| invalid())
    }
}

async fn ensure_installed(
    root: &Path,
    id: &str,
    version: &str,
    target: &BinaryTarget,
) -> Result<PathBuf, BinaryError> {
    let kind = ArchiveKind::of(&target.archive).ok_or_else(|| BinaryError::UnsupportedArchive {
        id: id.to_string(),
        archive: target.archive.clone(),
    })?;
    let cmd = relative_cmd(&target.cmd).ok_or_else(|| BinaryError::InvalidCommand {
        id: id.to_string(),
        cmd: target.cmd.clone(),
    })?;
    let dir = install_dir(root, id, version, target);
    let program = dir.join(&cmd);
    if program.is_file() {
        return Ok(program);
    }

    let _guard = INSTALL_LOCK.lock().await;
    if program.is_file() {
        return Ok(program);
    }

    let io = |source| BinaryError::Io {
        id: id.to_string(),
        source,
    };
    let agent_dir = dir.parent().unwrap_or(root);
    tokio::fs::create_dir_all(agent_dir).await.map_err(io)?;
    // Dropped (and deleted) on every early return below
    let staging = tempfile::Builder::new()
        .prefix(".staging-")
        .tempdir_in(agent_dir)
        .map_err(io)?;

    tracing::info!(id, version, archive = %target.archive, "Downloading ACP agent");
    let archive = staging.path().join("archive");
    download(id, target, &archive).await?;
    unpack_into(
        id,
        kind,
        archive,
        staging.path().join("unpacked"),
        &cmd,
        &dir,
    )
    .await?;
    tracing::info!(id, version, dir = %dir.display(), "Installed ACP agent");

    Ok(program)
}

/// Streams the archive to `dest`, failing unless it matches the registry's sha256.
async fn download(id: &str, target: &BinaryTarget, dest: &Path) -> Result<(), BinaryError> {
    let http = |source| BinaryError::Download {
        id: id.to_string(),
        source,
    };
    let io = |source| BinaryError::Io {
        id: id.to_string(),
        source,
    };

    let mut response = HTTP_CLIENT
        .get(&target.archive)
        .send()
        .await
        .and_then(|response| response.error_for_status())
        .map_err(http)?;
    let mut file = tokio::fs::File::create(dest).await.map_err(io)?;
    let mut hasher = Sha256::new();
    while let Some(chunk) = response.chunk().await.map_err(http)? {
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(io)?;
    }
    file.flush().await.map_err(io)?;

    let actual = format!("{:x}", hasher.finalize());
    let expected = target.sha256.trim();
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(BinaryError::ChecksumMismatch {
            id: id.to_string(),
            expected: expected.to_string(),
            actual,
        });
    }
    Ok(())
}

/// Unpacks `archive` into `staging`, then renames it to `dir` so `dir` only
/// ever holds a complete install.
async fn unpack_into(
    id: &str,
    kind: ArchiveKind,
    archive: PathBuf,
    staging: PathBuf,
    cmd: &Path,
    dir: &Path,
) -> Result<(), BinaryError> {
    let io = |source| BinaryError::Io {
        id: id.to_string(),
        source,
    };

    let unpacked = staging.clone();
    tokio::task::spawn_blocking(move || extract(kind, &archive, &unpacked))
        .await
        .map_err(|e| io(std::io::Error::other(e)))?
        .map_err(io)?;

    let program = staging.join(cmd);
    if !program.is_file() {
        return Err(BinaryError::MissingCommand {
            id: id.to_string(),
            cmd: cmd.display().to_string(),
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&program).map_err(io)?.permissions();
        permissions.set_mode(permissions.mode() | 0o111);
        std::fs::set_permissions(&program, permissions).map_err(io)?;
    }

    // A leftover without its command (the check in ensure_installed failed)
    match tokio::fs::remove_dir_all(dir).await {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io(e)),
        _ => {}
    }
    tokio::fs::rename(&staging, dir).await.map_err(io)
}

/// Both unpackers refuse entries that would land outside `dest`.
fn extract(kind: ArchiveKind, archive: &Path, dest: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dest)?;
    let file = std::fs::File::open(archive)?;
    match kind {
        ArchiveKind::TarGz => {
            let reader = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
            tar::Archive::new(reader).unpack(dest)
        }
        ArchiveKind::Zip => zip::ZipArchive::new(file)
            .and_then(|mut archive| archive.extract(dest))
            .map_err(std::io::Error::other),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn target(archive: &str, cmd: &str) -> BinaryTarget {
        BinaryTarget {
            archive: archive.to_string(),
            cmd: cmd.to_string(),
            args: vec!["acp".to_string()],
            env: Default::default(),
            sha256: "abc123".to_string(),
        }
    }

    fn kiro() -> BinaryTarget {
        target(
            "https://example.com/stable/3.0.0/kirocli.tar.gz",
            "./kirocli/bin/kiro-cli",
        )
    }

    /// A `.tar.gz` holding one executable script at `path`.
    fn tar_gz(path: &str) -> Vec<u8> {
        let encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        let body = b"#!/bin/sh\n";
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, path, &body[..]).unwrap();
        builder.into_inner().unwrap().finish().unwrap()
    }

    fn zip_file(path: &str) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        writer
            .start_file(path, zip::write::SimpleFileOptions::default())
            .unwrap();
        writer.write_all(b"#!/bin/sh\n").unwrap();
        writer.finish().unwrap().into_inner()
    }

    async fn unpack(
        kind: ArchiveKind,
        bytes: Vec<u8>,
        cmd: &str,
    ) -> (tempfile::TempDir, Result<(), BinaryError>) {
        let root = tempfile::tempdir().unwrap();
        let archive = root.path().join("archive");
        std::fs::write(&archive, bytes).unwrap();
        let result = unpack_into(
            "kiro",
            kind,
            archive,
            root.path().join("staging"),
            &relative_cmd(cmd).unwrap(),
            &root.path().join("installed"),
        )
        .await;
        (root, result)
    }

    #[test]
    fn archive_kind_needs_https_and_a_known_extension() {
        let kind = ArchiveKind::of;
        assert_eq!(
            kind("https://x.dev/a/kiro.tar.gz"),
            Some(ArchiveKind::TarGz)
        );
        assert_eq!(
            kind("https://x.dev/kiro.TGZ?sig=1"),
            Some(ArchiveKind::TarGz)
        );
        assert_eq!(kind("https://x.dev/kiro.zip"), Some(ArchiveKind::Zip));
        assert_eq!(kind("http://x.dev/kiro.zip"), None);
        assert_eq!(kind("https://x.dev/goose.tar.bz2"), None);
        assert_eq!(kind("https://x.dev/sigit-linux-amd64"), None);
        assert_eq!(kind("kiro.zip"), None);
    }

    #[test]
    fn relative_cmd_stays_inside_the_archive() {
        assert_eq!(
            relative_cmd("./kirocli/bin/kiro-cli"),
            Some(PathBuf::from("kirocli/bin/kiro-cli"))
        );
        assert_eq!(
            relative_cmd("./goose-package\\goose.exe"),
            Some(PathBuf::from("goose-package/goose.exe"))
        );
        for cmd in ["", ".", "/usr/bin/env", "../evil", "./a/../../evil"] {
            assert_eq!(relative_cmd(cmd), None, "{cmd}");
        }
    }

    #[test]
    fn host_target_picks_this_platform_only() {
        let platform = host_platform().expect("tests run on a supported platform");
        let mut targets = HashMap::from([("plan9-mips".to_string(), kiro())]);
        assert!(host_target(&targets).is_none());

        targets.insert(platform.clone(), kiro());
        assert!(host_target(&targets).is_some());

        // Listed for this platform, but not installable
        targets.insert(
            platform.clone(),
            target("https://example.com/goose.tar.bz2", "./goose"),
        );
        assert!(host_target(&targets).is_none());
        targets.insert(platform, target("https://example.com/a.zip", "../evil"));
        assert!(host_target(&targets).is_none());
    }

    #[test]
    fn install_dir_changes_with_version_and_archive() {
        let root = Path::new("/cache");
        let dir = install_dir(root, "kiro", "3.0.0", &kiro());
        assert_eq!(dir.parent().unwrap(), Path::new("/cache/KIRO"));
        let name = dir.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("v3.0.0-"), "{name}");

        assert_ne!(dir, install_dir(root, "kiro", "3.0.1", &kiro()));
        let republished = BinaryTarget {
            sha256: "def456".to_string(),
            ..kiro()
        };
        assert_ne!(dir, install_dir(root, "kiro", "3.0.0", &republished));

        // Registry text can't climb out of the agent's directory
        let hostile = install_dir(root, "../kiro", "../../3", &kiro());
        assert!(hostile.starts_with("/cache/___KIRO"));
        assert_eq!(hostile.components().count(), 4);
    }

    #[tokio::test]
    async fn unpacks_tar_gz_and_marks_the_command_executable() {
        let (root, result) = unpack(
            ArchiveKind::TarGz,
            tar_gz("kirocli/bin/kiro-cli"),
            "./kirocli/bin/kiro-cli",
        )
        .await;
        result.unwrap();

        let program = root.path().join("installed/kirocli/bin/kiro-cli");
        assert!(program.is_file());
        assert!(!root.path().join("staging").exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&program).unwrap().permissions().mode();
            assert_eq!(mode & 0o111, 0o111);
        }
    }

    #[tokio::test]
    async fn unpacks_zip() {
        let (root, result) = unpack(ArchiveKind::Zip, zip_file("bin/agent"), "bin/agent").await;
        result.unwrap();
        assert!(root.path().join("installed/bin/agent").is_file());
    }

    #[tokio::test]
    async fn unpack_replaces_a_broken_install() {
        let root = tempfile::tempdir().unwrap();
        let installed = root.path().join("installed");
        std::fs::create_dir_all(installed.join("stale")).unwrap();
        let archive = root.path().join("archive");
        std::fs::write(&archive, tar_gz("bin/agent")).unwrap();

        unpack_into(
            "kiro",
            ArchiveKind::TarGz,
            archive,
            root.path().join("staging"),
            Path::new("bin/agent"),
            &installed,
        )
        .await
        .unwrap();

        assert!(installed.join("bin/agent").is_file());
        assert!(!installed.join("stale").exists());
    }

    #[tokio::test]
    async fn unpack_fails_when_the_command_is_missing() {
        let (root, result) = unpack(ArchiveKind::TarGz, tar_gz("bin/other"), "bin/agent").await;
        assert!(matches!(result, Err(BinaryError::MissingCommand { .. })));
        assert!(!root.path().join("installed").exists());
    }

    #[tokio::test]
    async fn unpack_fails_on_a_corrupt_archive() {
        let (root, result) = unpack(ArchiveKind::Zip, b"not a zip".to_vec(), "bin/agent").await;
        assert!(matches!(result, Err(BinaryError::Io { .. })));
        assert!(!root.path().join("installed").exists());
    }

    #[tokio::test]
    async fn installed_agent_is_not_downloaded_again() {
        let root = tempfile::tempdir().unwrap();
        let program =
            install_dir(root.path(), "kiro", "3.0.0", &kiro()).join("kirocli/bin/kiro-cli");
        std::fs::create_dir_all(program.parent().unwrap()).unwrap();
        std::fs::write(&program, "").unwrap();

        // example.com has no such archive, so this passes only without a download
        let installed = ensure_installed(root.path(), "kiro", "3.0.0", &kiro())
            .await
            .unwrap();
        assert_eq!(installed, program);
    }

    #[cfg(not(windows))]
    #[test]
    fn quote_keeps_a_path_with_spaces_as_one_word() {
        let quoted = quote("kiro", Path::new("/Users/a b/kiro-cli")).unwrap();
        assert_eq!(shlex::split(&quoted).unwrap(), ["/Users/a b/kiro-cli"]);
    }
}
