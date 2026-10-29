#!/bin/bash
# monitor.sh OUT.tsv CPU... — every 5 s: time, 1-min load, per-CPU busy % and current MHz for the listed CPUs,
# and package/CPU temperatures. Runs until killed; used beside benchmark runs to spot thermal or load drift.
set -u
out=$1; shift
cpus=("$@")
read_stat() { awk -v c="cpu$1" '$1==c {idle=$5+$6; total=0; for (i=2;i<=NF;i++) total+=$i; print idle, total}' /proc/stat; }
declare -A idle0 total0
for c in "${cpus[@]}"; do read -r idle0[$c] total0[$c] <<<"$(read_stat "$c")"; done
printf "time\tload1" > "$out"
for c in "${cpus[@]}"; do printf "\tcpu%s_busy\tcpu%s_mhz" "$c" "$c" >> "$out"; done
printf "\tx86_pkg_temp\tTCPU\n" >> "$out"
while sleep 5; do
  line="$(date +%s)\t$(cut -d' ' -f1 /proc/loadavg)"
  for c in "${cpus[@]}"; do
    read -r idle total <<<"$(read_stat "$c")"
    busy=$(awk -v i=$((idle - idle0[$c])) -v t=$((total - total0[$c])) 'BEGIN { printf "%.0f", t ? 100 * (1 - i / t) : 0 }')
    idle0[$c]=$idle; total0[$c]=$total
    mhz=$(( $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq) / 1000 ))
    line="$line\t$busy\t$mhz"
  done
  temps=""
  for t in x86_pkg_temp TCPU; do
    zone=$(grep -l "^$t$" /sys/class/thermal/thermal_zone*/type | head -1)
    temps="$temps\t$(( $(cat "$(dirname "$zone")/temp") / 1000 ))"
  done
  printf "%b%b\n" "$line" "$temps" >> "$out"
done
