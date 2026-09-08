#!/usr/bin/env bash
set -euo pipefail

repo="zongshun21/-RF-DETR-"
tag="v1.0-insplad-baselines"
base="https://github.com/${repo}/releases/download/${tag}"

mkdir -p weights/releases
curl -fL -C - "${base}/rfdetr_s_insplad_640_best.pth" -o weights/releases/rfdetr_s_insplad_640_best.pth
curl -fL -C - "${base}/rfdetr_s_insplad_960_best.pth" -o weights/releases/rfdetr_s_insplad_960_best.pth
curl -fL "${base}/SHA256SUMS" -o weights/releases/SHA256SUMS
(cd weights/releases && sha256sum -c SHA256SUMS)
