#!/usr/bin/env bash
#
# download_data.sh -- fetch the Ausgrid selection and the pretrained MLflow runs.
#
#   scripts/download_data.sh [--archive PATH] [--url URL] [--dest DIR] [--dry-run]
#
# The authors currently host both pieces in one OneDrive folder:
#
#   https://1drv.ms/f/c/d5114f16b2467d66/ErohO9kQs3dEtu44wJrjXwMBcGFycoc8kBF6evk4bMvxhw?e=LStcCz
#
# OneDrive share links cannot be fetched non-interactively (they answer with an
# HTML page, not the archive), so this script does NOT pretend to download them:
# run it without arguments to get step-by-step manual instructions, download the
# folder as a single zip in a browser, and re-run with
#
#   scripts/download_data.sh --archive ~/Downloads/blstm-mionet-data.zip
#
# --url is for a future direct link (Zenodo, a GitHub release asset, ...) that
# curl can follow; the archive it points at is handled exactly like --archive.
#
# What happens with an archive:
#   * it is unpacked into a temporary directory (removed on exit),
#   * SHA-256 checksums are verified when scripts/checksums.sha256 lists any,
#   * the Ausgrid CSV folders are merged into data/Ausgrid/,
#   * the MLflow store is merged into ./mlruns (existing runs are never deleted),
#   * the script prints what landed where.
#
# This script needs no Python environment, only curl/unzip.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"

ONEDRIVE_URL="https://1drv.ms/f/c/d5114f16b2467d66/ErohO9kQs3dEtu44wJrjXwMBcGFycoc8kBF6evk4bMvxhw?e=LStcCz"
AUSGRID_URL="https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data"
CHECKSUM_FILE="$REPO_ROOT/scripts/checksums.sha256"
CONFIG="$REPO_ROOT/configs/ausgrid.yaml"

ARCHIVE=""
URL=""
DEST="$PWD"
DRY_RUN=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    cat <<'EOF'
Options:
  --archive PATH   zip archive that was already downloaded by hand
  --url URL        direct download link curl can follow (future Zenodo/GitHub release)
  --dest DIR       where data/Ausgrid and mlruns are created (default: repository root)
  --dry-run        show what would be copied, copy nothing
  -h, --help       show this message
EOF
}

hr() { printf '%s\n' "------------------------------------------------------------------------"; }
note() { printf '  %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --archive)
        [[ $# -ge 2 ]] || die "--archive needs a path"
        ARCHIVE="$2"
        shift 2
        ;;
    --archive=*)
        ARCHIVE="${1#--archive=}"
        shift
        ;;
    --url)
        [[ $# -ge 2 ]] || die "--url needs a URL"
        URL="$2"
        shift 2
        ;;
    --url=*)
        URL="${1#--url=}"
        shift
        ;;
    --dest)
        [[ $# -ge 2 ]] || die "--dest needs a directory"
        DEST="$2"
        shift 2
        ;;
    --dest=*)
        DEST="${1#--dest=}"
        shift
        ;;
    --dry-run)
        DRY_RUN=1
        shift
        ;;
    *)
        usage
        printf '\n'
        die "unknown argument '$1'"
        ;;
    esac
done

## The three CSV files configs/ausgrid.yaml expects, read straight out of the
## YAML so that this script stays in sync with the configuration.
expected_csv_paths() {
    if [[ -f "$CONFIG" ]]; then
        sed -n '/^[[:space:]]*csv_paths:[[:space:]]*$/,/^[[:space:]]*[a-zA-Z_]/ s/^[[:space:]]*-[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}[[:space:]]*$/\1/p' "$CONFIG"
    fi
}

print_manual_instructions() {
    hr
    printf 'Manual download (the OneDrive link cannot be fetched by curl)\n'
    hr
    cat <<EOF
  1. open this folder in a browser and sign in if you are asked to:

       $ONEDRIVE_URL

  2. use "Download" on the whole folder; OneDrive packs it into one zip file
     (a few GB: it holds the Ausgrid selection and the mlruns store with the
     pretrained registered models lorentz, pendulum and Ausgrid),

  3. re-run this script pointing at the file you downloaded:

       scripts/download_data.sh --archive ~/Downloads/<name>.zip

     or, once the archive is mirrored somewhere curl can reach,

       scripts/download_data.sh --url https://zenodo.org/.../blstm-mionet-data.zip

  The Ausgrid CSV files can also be rebuilt from the original source (free
  registration, the paper uses the "Solar home half-hour data" releases):

       $AUSGRID_URL
EOF
}

print_expected_layout() {
    printf '\n'
    hr
    printf 'Expected layout, relative to %s\n' "$DEST"
    hr
    local path
    while read -r path; do
        [[ -n "$path" ]] || continue
        if [[ -f "$DEST/$path" ]]; then
            note "present : $path"
        else
            note "missing : $path"
        fi
    done < <(expected_csv_paths)
    if [[ -d "$DEST/mlruns" ]]; then
        note "present : mlruns/ ($(find "$DEST/mlruns" -mindepth 1 -maxdepth 1 -type d | wc -l) top level entries)"
    else
        note "missing : mlruns/ (the pretrained runs; training re-creates it)"
    fi
    printf '\n'
    note "Ausgrid original source: $AUSGRID_URL"
    note "Authors' OneDrive folder: $ONEDRIVE_URL"
}

## ---------------------------------------------------------------------------
## No archive at all: explain what to do and stop.
## ---------------------------------------------------------------------------
if [[ -z "$ARCHIVE" && -z "$URL" ]]; then
    print_manual_instructions
    print_expected_layout
    printf '\n'
    note "Nothing was downloaded; re-run with --archive PATH once you have the zip."
    exit 0
fi

WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

## ---------------------------------------------------------------------------
## Fetch --url into the temporary directory.
## ---------------------------------------------------------------------------
if [[ -n "$URL" ]]; then
    if [[ "$URL" == *"1drv.ms"* || "$URL" == *"onedrive.live.com"* || "$URL" == *"sharepoint.com"* ]]; then
        warn "OneDrive share links cannot be downloaded non-interactively."
        print_manual_instructions
        exit 1
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required for --url"
    ARCHIVE="$WORK_DIR/download.zip"
    printf 'Downloading %s\n' "$URL"
    if ! curl -fL --retry 3 --retry-delay 2 -o "$ARCHIVE" "$URL"; then
        warn "the download failed."
        print_manual_instructions
        exit 1
    fi
fi

[[ -f "$ARCHIVE" ]] || {
    warn "no such archive: $ARCHIVE"
    print_manual_instructions
    exit 1
}

## ---------------------------------------------------------------------------
## Unpack.
## ---------------------------------------------------------------------------
EXTRACT_DIR="$WORK_DIR/extracted"
mkdir -p "$EXTRACT_DIR"
printf 'Unpacking %s\n' "$ARCHIVE"
if command -v unzip >/dev/null 2>&1; then
    unzip -q -o "$ARCHIVE" -d "$EXTRACT_DIR"
elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$ARCHIVE" "$EXTRACT_DIR"
else
    die "neither 'unzip' nor 'python3' is available to unpack $ARCHIVE"
fi

## ---------------------------------------------------------------------------
## Checksums (the file ships empty; the maintainer fills it after publishing).
## ---------------------------------------------------------------------------
if [[ -f "$CHECKSUM_FILE" ]] && grep -qvE '^[[:space:]]*(#.*)?$' "$CHECKSUM_FILE"; then
    printf '\nVerifying SHA-256 checksums from %s\n' "$CHECKSUM_FILE"
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to verify $CHECKSUM_FILE"
    (cd "$EXTRACT_DIR" && sha256sum --check --ignore-missing "$CHECKSUM_FILE") ||
        die "checksum verification failed; the archive looks incomplete or corrupted"
    note "checksums OK"
else
    warn "no checksums listed in scripts/checksums.sha256; skipping verification"
fi

## ---------------------------------------------------------------------------
## Merge the payload into the destination.
## ---------------------------------------------------------------------------
## copy_merge SRC DST -- copy without ever replacing or deleting existing files.
copy_merge() {
    local src="$1" dst="$2"
    if [[ $DRY_RUN -eq 1 ]]; then
        note "would copy $src -> $dst"
        return 0
    fi
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --ignore-existing "$src/" "$dst/"
    else
        cp -rn "$src/." "$dst/"
    fi
}

AUSGRID_DEST="$DEST/data/Ausgrid"
MLRUNS_DEST="$DEST/mlruns"
COPIED_AUSGRID=0
COPIED_MLRUNS=0

printf '\n'
hr
printf 'Merging the archive into %s\n' "$DEST"
hr

## 1. An "Ausgrid" directory, if the archive has one, otherwise the individual
##    "Solar home half-hour data ..." folders, otherwise loose CSV files.
while IFS= read -r directory; do
    note "Ausgrid folder: ${directory#"$EXTRACT_DIR"/}"
    copy_merge "$directory" "$AUSGRID_DEST"
    COPIED_AUSGRID=1
done < <(find "$EXTRACT_DIR" -type d -name "Ausgrid" -prune)

if [[ $COPIED_AUSGRID -eq 0 ]]; then
    while IFS= read -r directory; do
        note "Ausgrid folder: ${directory#"$EXTRACT_DIR"/}"
        copy_merge "$directory" "$AUSGRID_DEST/$(basename "$directory")"
        COPIED_AUSGRID=1
    done < <(find "$EXTRACT_DIR" -type d -name "*Solar home half-hour data*" -prune)
fi

if [[ $COPIED_AUSGRID -eq 0 ]]; then
    while IFS= read -r csv; do
        note "Ausgrid CSV: ${csv#"$EXTRACT_DIR"/}"
        if [[ $DRY_RUN -eq 0 ]]; then
            mkdir -p "$AUSGRID_DEST"
            cp -n "$csv" "$AUSGRID_DEST/"
        fi
        COPIED_AUSGRID=1
    done < <(find "$EXTRACT_DIR" -type f -iname "*Solar home electricity data*.csv")
fi

[[ $COPIED_AUSGRID -eq 1 ]] || warn "no Ausgrid CSV files were found in the archive"

## 2. The MLflow store, merged so that local runs survive.
while IFS= read -r directory; do
    note "MLflow store  : ${directory#"$EXTRACT_DIR"/}"
    copy_merge "$directory" "$MLRUNS_DEST"
    COPIED_MLRUNS=1
done < <(find "$EXTRACT_DIR" -type d -name "mlruns" -prune)

[[ $COPIED_MLRUNS -eq 1 ]] || warn "no mlruns/ directory was found in the archive"

if [[ $DRY_RUN -eq 1 ]]; then
    printf '\n'
    note "--dry-run: nothing was written."
fi

print_expected_layout

printf '\n'
if [[ $COPIED_MLRUNS -eq 1 && $DRY_RUN -eq 0 ]]; then
    note "The runs were written on another machine and MLflow stores absolute paths,"
    note "so point the store at its new location once, then use the pretrained models:"
    note "  blstm-mionet relocate-mlruns $MLRUNS_DEST"
    note "  blstm-mionet infer --config configs/lorentz.yaml --model models:/lorentz/latest"
    note "  (set MLFLOW_TRACKING_URI=$MLRUNS_DEST when running from another directory)"
fi
