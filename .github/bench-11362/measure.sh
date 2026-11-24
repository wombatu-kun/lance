#!/usr/bin/env bash
set -euo pipefail
kit=$(realpath .github/bench-11362)
out="$RUNNER_TEMP/arm64-jni-measurements"
mkdir "$out"
previous_profile=''
finish() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -n $previous_profile ]]; then
        sudo -n powerprofilesctl set "$previous_profile" || status=1
    fi
    printf '%s\n' "$status" > "$out/exit-code"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

[[ $(uname -m) == aarch64 ]]
case "$(docker info --format '{{.Architecture}}')" in arm64|aarch64) ;; *) exit 2 ;; esac
lscpu > "$out/cpu.txt"
cat /proc/meminfo > "$out/memory.txt"
docker info > "$out/docker.txt"
if profile=$(powerprofilesctl get 2>/dev/null); then
    previous_profile=$profile
    sudo -n powerprofilesctl set performance
    [[ $(powerprofilesctl get) == performance ]]
fi
source_rev=$(git -C pinned-source rev-parse HEAD)
[[ $source_rev == f9eb14ace1f0b153e9092087f9a1026e0182d11d ]]
echo "$source_rev" > "$out/source-revision.txt"
image=debian:10@sha256:58ce6f1271ae1c8a2006ff7d3e54e9874d839f573d8009c20154ad0f2fb0a225
docker pull --platform linux/arm64 "$image"
arms=(baseline candidate)
if ((GITHUB_RUN_ATTEMPT % 2 == 0)); then arms=(candidate baseline); fi
printf '%s\n' "${arms[@]}" | tee "$out/arm-order.txt"
for arm in "${arms[@]}"; do
    src="$RUNNER_TEMP/source-$arm"
    results="$out/$arm"
    mkdir "$src" "$results"
    git -C pinned-source archive HEAD | tar -x -C "$src"
    if [[ $arm == candidate ]]; then
        git -C "$src" apply --no-index "$kit/candidate.patch"
        expected=10c24df46ff587a487d72e60bc1431ba7b113bac3f29017c88cf759b07e1ee5e
    else
        expected=23aaf1b3221130799485afc0cb51ae66c5fea5495ded27a5beb64b9019094a7d
    fi
    sha256sum "$src/rust/lance/src/session.rs" > "$results/session-source.txt"
    [[ $(cut -d ' ' -f1 "$results/session-source.txt") == "$expected" ]]
    if [[ -n $previous_profile ]]; then
        powerprofilesctl get > "$results/power-profile.txt"
        [[ $(cat "$results/power-profile.txt") == performance ]]
    else
        echo 'Host power profile unavailable in VM. Preliminary measurement.' > "$results/power-profile.txt"
    fi
    for path in /sys/devices/system/cpu/cpu0/cpufreq/{scaling_governor,energy_performance_preference} /sys/firmware/acpi/platform_profile; do
        if [[ -r $path ]]; then
            setting=$(cat "$path")
            echo "$setting" > "$results/${path##*/}.txt"
            if [[ -z $previous_profile && $setting != performance ]]; then
                echo "Visible power setting $path=$setting; performance is required." >&2
                exit 2
            fi
        fi
    done
    cat "$results/power-profile.txt"
    date -u --iso-8601=seconds > "$results/started.txt"
    sudo -n df -h "$src" "$(docker info --format '{{.DockerRootDir}}')" > "$results/disk-before.txt"
    grep -E '^(pswpin|pswpout|oom_kill) ' /proc/vmstat > "$results/vm-before.txt"
    status=0
    docker run --rm --platform linux/arm64 \
        --mount "type=bind,source=$src,target=/workspace" \
        --mount "type=volume,target=/workspace/java/lance-jni/target" \
        --mount "type=bind,source=$kit/build.sh,target=/measure-build.sh,readonly" \
        --mount "type=bind,source=$results,target=/results" \
        -w /workspace "$image" bash /measure-build.sh 2>&1 | tee "$results/build.log" || status=$?
    echo "$status" > "$results/exit-code"
    date -u --iso-8601=seconds > "$results/finished.txt"
    sudo -n df -h "$src" "$(docker info --format '{{.DockerRootDir}}')" > "$results/disk-after.txt"
    grep -E '^(pswpin|pswpout|oom_kill) ' /proc/vmstat > "$results/vm-after.txt"
    if ((status != 0)); then exit "$status"; fi
    if [[ -n $previous_profile ]]; then [[ $(powerprofilesctl get) == performance ]]; fi
    echo "$arm"
    cat "$results/session-source.txt" "$results/rustc.txt" "$results/times.txt"
done
for item in inputs.txt rustc.txt cargo.txt clang.txt packages.txt power-profile.txt; do
    cmp "$out/baseline/$item" "$out/candidate/$item"
done
for item in online-cpus cpu.max cpuset.cpus.effective memory.max cpu.cfs_quota_us cpu.cfs_period_us scaling_governor energy_performance_preference platform_profile; do
    if [[ -e $out/baseline/$item.txt || -e $out/candidate/$item.txt ]]; then
        cmp "$out/baseline/$item.txt" "$out/candidate/$item.txt"
    fi
done
