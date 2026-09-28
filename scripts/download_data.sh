#!/usr/bin/env bash
#
# download_data.sh -- fetch the Ausgrid CSV files and the pretrained MLflow runs.
#
#   scripts/download_data.sh [--archive PATH]... [--url URL]... [--dest DIR] [--dry-run]
#
# Both live in the GitHub release "data-v1.0" of this repository:
#
#   Ausgrid.zip  the three Ausgrid "Solar home half-hour data" CSV files
#   mlruns.zip   the MLflow store of the paper, with the registered models
#                lorentz, pendulum and Ausgrid
#
# Run without arguments to download both.  --archive takes zips downloaded by
# hand (for example from the release page in a browser) and --url other direct
# links; both may be repeated.  For every archive:
#   * its SHA-256 is checked against scripts/checksums.sha256 (by file name),
#   * it is unpacked into a temporary directory (removed on exit),
#   * data/Ausgrid/ and mlruns/ are merged into the destination without ever
#     replacing or deleting existing files.
# Afterwards `blstm-mionet relocate-mlruns` points the store at its new location
# (MLflow records absolute paths); when the command is not available the script
# prints it instead.
#
# Only curl and unzip (or python3) are needed.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"

RELEASE_URL="https://github.com/moodykong/bayesian-lstm-mionet/releases/download/data-v1.0"
RELEASE_PAGE="https://github.com/moodykong/bayesian-lstm-mionet/releases/tag/data-v1.0"
RELEASE_ASSETS=(Ausgrid.zip mlruns.zip)
# Ausgrid no longer hosts the dataset page; this is the paper describing the data.
AUSGRID_URL="https://doi.org/10.1080/14786451.2015.1100196"
CHECKSUM_FILE="$REPO_ROOT/scripts/checksums.sha256"
CONFIG="$REPO_ROOT/configs/ausgrid.yaml"

ARCHIVES=()
URLS=()
DEST="$PWD"
DRY_RUN=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
    cat <<'EOF'
Options:
  --archive PATH   a zip downloaded by hand (repeatable)
  --url URL        a direct download link curl can follow (repeatable)
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
        ARCHIVES+=("$2")
        shift 2
        ;;
    --archive=*)
        ARCHIVES+=("${1#--archive=}")
        shift
        ;;
    --url)
        [[ $# -ge 2 ]] || die "--url needs a URL"
        URLS+=("$2")
        shift 2
        ;;
    --url=*)
        URLS+=("${1#--url=}")
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
    printf 'Manual download\n'
    hr
    cat <<EOF
  Download Ausgrid.zip and mlruns.zip from the release page in a browser:

       $RELEASE_PAGE

  and re-run this script with the files you downloaded:

       scripts/download_data.sh --archive ~/Downloads/Ausgrid.zip --archive ~/Downloads/mlruns.zip

  The Ausgrid data are described in

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
    note "Ausgrid dataset description: $AUSGRID_URL"
    note "Release with both archives : $RELEASE_PAGE"
}

WORK_DIR="$(mktemp -d)"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

## ---------------------------------------------------------------------------
## Without --archive or --url, fetch the release assets.
## ---------------------------------------------------------------------------
if [[ ${#ARCHIVES[@]} -eq 0 && ${#URLS[@]} -eq 0 ]]; then
    for asset in "${RELEASE_ASSETS[@]}"; do
        URLS+=("$RELEASE_URL/$asset")
    done
fi

if [[ ${#URLS[@]} -gt 0 ]]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to download; use --archive instead"
    mkdir -p "$WORK_DIR/downloads"
    for url in "${URLS[@]}"; do
        target="$WORK_DIR/downloads/$(basename "${url%%\?*}")"
        printf 'Downloading %s\n' "$url"
        if ! curl -fL --retry 3 --retry-delay 2 -o "$target" "$url"; then
            warn "the download of $url failed."
            print_manual_instructions
            exit 1
        fi
        ARCHIVES+=("$target")
    done
fi

## ---------------------------------------------------------------------------
## Check and unpack every archive.
## ---------------------------------------------------------------------------
## verify_checksum ZIP -- compare with the line for its file name, if any.
verify_checksum() {
    local zip="$1" name expected actual
    name="$(basename "$zip")"
    expected="$(awk -v n="$name" '$2 == n || $2 == "*"n {print $1}' "$CHECKSUM_FILE" 2>/dev/null | head -n 1)"
    if [[ -z "$expected" ]]; then
        warn "no checksum listed for $name in scripts/checksums.sha256; not verified"
        return 0
    fi
    command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to verify $name"
    actual="$(sha256sum "$zip" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] ||
        die "SHA-256 mismatch for $name: the file is incomplete or not the published one"
    note "checksum OK: $name"
}

EXTRACT_DIR="$WORK_DIR/extracted"
mkdir -p "$EXTRACT_DIR"
for archive in "${ARCHIVES[@]}"; do
    [[ -f "$archive" ]] || {
        warn "no such archive: $archive"
        print_manual_instructions
        exit 1
    }
    verify_checksum "$archive"
    printf 'Unpacking %s\n' "$archive"
    if command -v unzip >/dev/null 2>&1; then
        unzip -q -o "$archive" -d "$EXTRACT_DIR"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m zipfile -e "$archive" "$EXTRACT_DIR"
    else
        die "neither 'unzip' nor 'python3' is available to unpack $archive"
    fi
done

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
    elif cp --help 2>/dev/null | grep -q -- "--update\[=UPDATE\]"; then
        cp -R --update=none "$src/." "$dst/" # GNU coreutils >= 9.3
    else
        cp -Rn "$src/." "$dst/"
    fi
}

AUSGRID_DEST="$DEST/data/Ausgrid"
MLRUNS_DEST="$DEST/mlruns"
COPIED_AUSGRID=0
COPIED_MLRUNS=0

printf '\n'
hr
printf 'Merging the archives into %s\n' "$DEST"
hr

## 1. An "Ausgrid" directory, if the archive has one, otherwise the individual
##    "Solar home half-hour data ..." folders, otherwise loose CSV files.
while IFS= read -r directory; do
    note "Ausgrid folder: ${directory#"$EXTRACT_DIR"/}"
    copy_merge "$directory" "$AUSGRID_DEST"
    COPIED_AUSGRID=1
done < <(find "$EXTRACT_DIR" -type d -name "mlruns" -prune -o -type d -name "Ausgrid" -print -prune)

if [[ $COPIED_AUSGRID -eq 0 ]]; then
    while IFS= read -r directory; do
        note "Ausgrid folder: ${directory#"$EXTRACT_DIR"/}"
        copy_merge "$directory" "$AUSGRID_DEST/$(basename "$directory")"
        COPIED_AUSGRID=1
    done < <(find "$EXTRACT_DIR" -type d -name "mlruns" -prune -o -type d -name "*Solar home half-hour data*" -print -prune)
fi

if [[ $COPIED_AUSGRID -eq 0 ]]; then
    while IFS= read -r csv; do
        note "Ausgrid CSV: ${csv#"$EXTRACT_DIR"/}"
        if [[ $DRY_RUN -eq 0 ]]; then
            mkdir -p "$AUSGRID_DEST"
            [[ -e "$AUSGRID_DEST/$(basename "$csv")" ]] || cp "$csv" "$AUSGRID_DEST/"
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
    ## MLflow records absolute paths and these runs were written elsewhere.
    CLI=()
    if [[ -n "${BLSTM_MIONET_CMD:-}" ]]; then
        read -r -a CLI <<<"${BLSTM_MIONET_CMD}"
    elif command -v blstm-mionet >/dev/null 2>&1; then
        CLI=(blstm-mionet)
    elif command -v uv >/dev/null 2>&1 && [[ -d "$REPO_ROOT/.venv" ]]; then
        CLI=(uv run --no-sync --project "$REPO_ROOT" blstm-mionet)
    fi
    if [[ ${#CLI[@]} -gt 0 ]] && MLFLOW_DISABLE_AGENT_HINT=1 "${CLI[@]}" relocate-mlruns "$MLRUNS_DEST"; then
        note "The pretrained models are ready, for example:"
    else
        note "MLflow records absolute paths; point the store at its new location once:"
        note "  blstm-mionet relocate-mlruns $MLRUNS_DEST"
        note "then use the pretrained models, for example:"
    fi
    note "  blstm-mionet infer --config configs/lorentz.yaml --model models:/lorentz/latest"
    note "  (set MLFLOW_TRACKING_URI=$MLRUNS_DEST when running from another directory)"
fi
