#!/usr/bin/env bash
set -euo pipefail

image="${1:-grasp-eval:local}"
code_commit="$(git rev-parse HEAD)"

docker build \
  --platform linux/amd64 \
  --file Dockerfile.eval \
  --build-arg "CODE_COMMIT=${code_commit}" \
  --tag "${image}" \
  .

docker run --rm --platform linux/amd64 "${image}"
