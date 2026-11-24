#!/bin/bash
set -e
read -r container_start _ < /proc/uptime
capture_resources() {
    local code=$?
    set +ex
    df -h /workspace/java/lance-jni/target / > /results/container-disk.txt
    for item in /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events; do
        if [[ -r $item ]]; then cat "$item" > "/results/${item##*/}.txt"; fi
    done
    exit "$code"
}
trap capture_resources EXIT
set -ex
# Update sources.list to use archive repositories for Debian 10 (EOL)
echo 'deb http://archive.debian.org/debian/ buster main' > /etc/apt/sources.list
echo 'deb http://archive.debian.org/debian-security buster/updates main' >> /etc/apt/sources.list
echo 'deb http://archive.debian.org/debian/ buster-updates main' >> /etc/apt/sources.list
apt-get update

DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --assume-yes \
  apt-transport-https \
  ca-certificates \
  curl \
  gpg \
  bash \
  less \
  openssl \
  libssl-dev \
  pkg-config \
  libsqlite3-dev \
  libsqlite3-0 \
  libreadline-dev \
  git \
  cmake \
  dh-autoreconf \
  clang \
  g++ \
  libc++-dev \
  libc++abi-dev \
  libprotobuf-dev \
  libncurses5-dev \
  libncursesw5-dev \
  libudev-dev \
  libhidapi-dev \
  zip \
  unzip

# https://github.com/databendlabs/databend/issues/8035
PROTOC_ZIP=protoc-3.15.0-linux-aarch_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.15.0/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d /usr/local
rm -f $PROTOC_ZIP
protoc --version

curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
source $HOME/.cargo/env
cargo --version

cd java/lance-jni

# https://github.com/rustls/rustls/issues/1967
export CC=clang
export CXX=clang++
ldd --version


set +x
uname -a > /results/kernel.txt
getconf _NPROCESSORS_ONLN > /results/online-cpus.txt
cat /proc/self/cgroup > /results/cgroup.txt
for item in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us; do
    if [[ -r $item ]]; then cat "$item" > "/results/${item##*/}.txt"; fi
done
rustc -Vv > /results/rustc.txt
cargo -V > /results/cargo.txt
clang --version > /results/clang.txt
dpkg-query -W > /results/packages.txt
sha256sum Cargo.lock ../../.cargo/config.toml ../../rust-toolchain.toml > /results/inputs.txt
test ! -e target/release
read -r cargo_start _ < /proc/uptime
set -x
cargo build --release --locked
set +x
read -r finished _ < /proc/uptime
test -s target/release/liblance_jni.so
readelf -h target/release/liblance_jni.so > /results/elf.txt
grep -q 'Machine:.*AArch64' /results/elf.txt
grep -q 'Type:.*DYN' /results/elf.txt
awk -v start="$container_start" -v cargo="$cargo_start" -v end="$finished" 'BEGIN {printf "setup_seconds=%.2f\ncargo_seconds=%.2f\ncontainer_to_library_seconds=%.2f\n", cargo-start, end-cargo, end-start}' > /results/times.txt
sha256sum target/release/liblance_jni.so > /results/library.sha256
