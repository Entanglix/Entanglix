#!/usr/bin/env bash
# Install Entanglix for the current user, without root:
#   curl -fsSL https://raw.githubusercontent.com/Entanglix/Entanglix/vX.Y.Z/install.sh | bash
# Everything runs inside main, called on the last line, so a truncated download runs nothing.
# Written for bash 3.2, which macOS ships.

main() {
  set -eu
  version=0.1.0
  uv_version=0.12.0
  repository=${ENTANGLIX_INSTALL_REPOSITORY:-https://github.com/Entanglix/Entanglix.git}
  ref=${ENTANGLIX_INSTALL_REF:-v$version}

  system=$(uname -s)
  case "$system $(uname -m)" in
    "Linux x86_64" | "Linux amd64")
      uv_target=x86_64-unknown-linux-musl
      uv_sha256=3340a9d8cffc4d801bc1a7459ebfaf5790c79400720d9b6963d806f058526684
      ;;
    "Linux aarch64" | "Linux arm64")
      uv_target=aarch64-unknown-linux-musl
      uv_sha256=936fbbf20188a2b1c66bce3dca3f4009a5c9cdf12bb2bbd084e71926f75d6a15
      ;;
    "Darwin arm64")
      uv_target=aarch64-apple-darwin
      uv_sha256=2b9e582af54f84fa50c115427451a6c13e80f43b52f8282b8af5791077317bbf
      ;;
    *)
      fail "Entanglix supports Linux on x86_64 and aarch64 and macOS on Apple Silicon," \
        "not $system on $(uname -m)"
      ;;
  esac
  # macOS ships shasum and perl in place of sha256sum and flock.
  if [ "$system" = Darwin ]; then
    tools="git curl tar shasum perl"
  else
    tools="git curl tar sha256sum flock"
  fi
  for tool in $tools; do
    command -v "$tool" >/dev/null 2>&1 || fail "Entanglix needs $tool; install it and run this again"
  done

  data=${XDG_DATA_HOME:-$HOME/.local/share}/entanglix
  target=$data/$version
  state=$target/.installation
  uv=$state/bin/uv
  mkdir -p "$data"
  # One install or removal at a time, so none sees another's directories half made.
  exec 9>>"$data/.lock"
  if [ "$system" = Darwin ]; then
    # The same lock as flock takes, on the descriptor this shell keeps open.
    perl -MFcntl=:flock -e 'open(my $lock, ">>&=", 9) or exit 1; flock($lock, LOCK_EX | LOCK_NB) or exit 1'
  else
    flock -n 9
  fi || fail "another Entanglix install or removal is running; run this again after it ends"

  if [ ! -f "$state/installation.toml" ]; then
    if [ -e "$target" ] && [ ! -d "$target/.git" ]; then
      fail "$target exists but is not an Entanglix clone; move it away and run this again"
    fi
    if [ ! -d "$target/.git" ]; then
      rm -rf "$target.partial"
      echo "Cloning Entanglix $version" >&2
      git init --quiet "$target.partial"
      git -C "$target.partial" remote add origin "$repository"
      git -C "$target.partial" fetch --quiet --depth 1 origin "$ref"
      git -C "$target.partial" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
      mv "$target.partial" "$target"
    fi
    grep -q "^version = \"$version\"" "$target/pyproject.toml" ||
      fail "$ref of $repository is not Entanglix $version"
    if [ ! -x "$uv" ]; then
      echo "Downloading uv $uv_version" >&2
      mkdir -p "$state/bin"
      archive=$state/uv.tar.gz
      curl -fsSL -o "$archive" \
        "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$uv_target.tar.gz"
      if [ "$system" = Darwin ]; then
        digest=$(shasum -a 256 "$archive")
      else
        digest=$(sha256sum "$archive")
      fi
      [ "${digest%% *}" = "$uv_sha256" ] || fail "the uv download does not match its checksum"
      tar -xzf "$archive" -C "$state/bin" --strip-components=1 "uv-$uv_target/uv"
      rm -f "$archive"
    fi
  fi
  # Entanglix's own uv cache, shared by its versions; removing the last version deletes it.
  cache=$data/cache

  if [ ! -f "$state/installation.toml" ]; then
    echo "Creating the Entanglix environment" >&2
    (
      # Inherited variables cannot redirect this uv; network settings pass.
      for name in $(compgen -e); do
        case $name in
          UV_NATIVE_TLS | UV_HTTP_TIMEOUT | UV_HTTP_RETRIES | UV_CONCURRENT_DOWNLOADS) ;;
          UV_CONCURRENT_BUILDS | UV_CONCURRENT_INSTALLS) ;;
          UV | UV_* | VIRTUAL_ENV | PYTHONPATH | PYTHONHOME) unset "$name" ;;
        esac
      done
      export UV_CACHE_DIR=$cache UV_PYTHON_INSTALL_DIR=$state/python
      export UV_PROJECT_ENVIRONMENT=$target/.venv
      # uv looks for .python-version from the working directory, so the pin is explicit.
      "$uv" sync --quiet --project "$target" --locked --no-config --no-dev --managed-python \
        --python "$(cat "$target/.python-version")" --extra mcp
    )
  fi
  # The rest is Python: the base job environment, the completion record and the launcher.
  env -u PYTHONPATH -u PYTHONHOME "$target/.venv/bin/python" -m entanglix.installation install \
    --uv "$uv" --cache "$cache" --bin "$HOME/.local/bin"
}

fail() {
  echo "entanglix install: $*" >&2
  exit 1
}

main "$@"
