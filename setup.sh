#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get -y install --no-install-recommends \
            git \
            make \
            cmake \
            build-essential \
            python3-dev \
            python3-pip \
            libssl-dev \
            zlib1g-dev \
            libbz2-dev \
            libreadline-dev \
            libsqlite3-dev \
            liblzma-dev \
            libffi-dev \
            curl \
            ca-certificates

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "Setup completed."