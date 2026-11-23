#!/usr/bin/env bash
# Launch the sinusoidal partial modeler GUI.
#
# On a normal desktop with PySide6 installed system-wide this is just
# `python3 -m sinmodel.gui`.  The extra library path below only matters on
# minimal containers where the Qt6 runtime libraries (EGL/GL/PulseAudio) were
# unpacked into a local prefix instead of being installed via the system
# package manager; it is harmless when that directory does not exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_LIB_DIR="${SINMODEL_EXTRA_LIBS:-/tmp/syslibs}"

if [ -d "${EXTRA_LIB_DIR}/usr/lib/aarch64-linux-gnu" ]; then
  export LD_LIBRARY_PATH="$(find "${EXTRA_LIB_DIR}" -type d \
    \( -name 'aarch64-linux-gnu' -o -name 'pulseaudio' \) -printf '%p:' \
    2>/dev/null)${LD_LIBRARY_PATH:-}"
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-}"
exec python3 -m sinmodel.gui "$@"
