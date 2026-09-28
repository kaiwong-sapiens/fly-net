#!/usr/bin/env bash
# Fetch the FlyWire (v783) connectivity used by Shiu et al. 2024 and the FlyWire cell-type annotations.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data && cd data
[ -d Drosophila_brain_model ] || git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git
[ -d flywire_annotations ] || git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git
echo "data ready in $(pwd)"
