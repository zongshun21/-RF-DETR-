#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights/releases
cat model_weights/chunks/rfdetr_s_insplad_640_best.pth.part-* > weights/releases/rfdetr_s_insplad_640_best.pth
cat model_weights/chunks/rfdetr_s_insplad_960_best.pth.part-* > weights/releases/rfdetr_s_insplad_960_best.pth
cp model_weights/SHA256SUMS weights/releases/SHA256SUMS
(cd weights/releases && sha256sum -c SHA256SUMS)
