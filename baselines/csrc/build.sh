#!/bin/bash
# Build the DMT and homotopy-warping kernels for the active python.
#
#   bash baselines/csrc/build.sh [build_dir]     (default: baselines/csrc/build)
#
# A build outside the default directory is found through TOPO_BASELINES_BUILD.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${1:-$HERE/build}"
PYTHON="${PYTHON:-python}"
INC="$("$PYTHON" -c 'import sysconfig, pybind11; print("-I" + sysconfig.get_paths()["include"], "-I" + pybind11.get_include())')"
EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
mkdir -p "$OUT_DIR"
for name in topo_baselines_core dmt3d_core; do
    g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden -pthread $INC \
        "$HERE/$name.cpp" -o "$OUT_DIR/$name$EXT.tmp"
    mv -f "$OUT_DIR/$name$EXT.tmp" "$OUT_DIR/$name$EXT"
    echo "built $OUT_DIR/$name$EXT"
done
