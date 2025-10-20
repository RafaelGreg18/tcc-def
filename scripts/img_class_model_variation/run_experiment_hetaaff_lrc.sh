#!/usr/bin/env bash
# Execução com alta prioridade (CPU/I/O) e afinidade de CPU para jobs que usam CPU e GPU.
# Modos:
#   HIGH_PRIO_MODE=batch   (padrão; seguro: cgroup weights + ionice + taskset; tenta nice -5)
#   HIGH_PRIO_MODE=service (service transitório; idem acima, BLOQUEIA até terminar)
#   HIGH_PRIO_MODE=rt      (tempo real RR com chrt; perigoso; requer root/sudo)
#
# Exemplos:
#   ./seu_script.sh 0,1
#   ALLOWED_CPUS=0-7 ./seu_script.sh 0
#   HIGH_PRIO_MODE=service ./seu_script.sh 0
#   SUDO=sudo HIGH_PRIO_MODE=batch ./seu_script.sh 0
#   sudo -E HIGH_PRIO_MODE=rt RT_PRIO=70 ./seu_script.sh 0

# --- Requer bash (evita rodar acidentalmente com /bin/sh) ---
if [ -z "${BASH_VERSION:-}" ]; then
  echo "Este script requer bash. Rode com: bash $0 <CUDA_DEVICE_IDS>" >&2
  exit 2
fi

set -euo pipefail

# ------------------------ Knobs (sobreponha por env se quiser) -----------------
: "${HIGH_PRIO_MODE:=batch}"         # batch | service | rt
: "${CPU_WEIGHT:=10000}"             # 1..10000 (cgroup v2)
: "${IO_WEIGHT:=1000}"               # 10..1000  (cgroup v2)
: "${RT_PRIO:=70}"                   # prioridade RR 1..99 (modo rt)

# ------------------------ Validação de ALLOWED_CPUS ----------------------------
build_default_cpuset() { echo "0-$(( $(nproc) - 1 ))"; }
is_valid_cpuset() {
  # Aceita: "0-7", "0,2,4,6", "0-3,8-11", etc.
  printf '%s' "${1:-}" | grep -Eq '^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$'
}

: "${ALLOWED_CPUS:=$(build_default_cpuset)}"
if ! is_valid_cpuset "$ALLOWED_CPUS"; then
  echo "Aviso: ALLOWED_CPUS inválido: '$ALLOWED_CPUS' — usando $(build_default_cpuset)" >&2
  ALLOWED_CPUS="$(build_default_cpuset)"
fi
export ALLOWED_CPUS

# ------------------------ Entrada: IDs de GPU / CUDA ---------------------------
if [ -z "${1:-}" ]; then
  echo "Usage: $0 <CUDA_DEVICE_IDS>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"
: "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:=75}"
: "${CUDA_MPS_PIPE_DIRECTORY:=$HOME/.nvidia-mps/pipe}"
: "${CUDA_MPS_LOG_DIRECTORY:=$HOME/.nvidia-mps/log}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# ------------------------ Ray config -------------------------------------------
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

# ------------------------ Sobe MPS (não root, sem warnings) --------------------
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  if ! pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
    nvidia-cuda-mps-control -d || true
  fi
fi

# ------------------------ Wrapper de alta prioridade ---------------------------
run_hp() {
  # Decide se roda no user manager ou system manager (via sudo)
  local run_sysd=(systemd-run --user)
  if [ -n "${SUDO:-}" ]; then
    run_sysd=("$SUDO" systemd-run)    # sem --user quando via sudo (system manager)
  fi

  # Vars de ambiente para dentro do systemd-run
  local env_flags=(
    "--setenv=ALLOWED_CPUS=${ALLOWED_CPUS}"
    "--setenv=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    "--setenv=CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-}"
    "--setenv=CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-}"
    "--setenv=CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-}"
    "--setenv=RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-}"
    "--setenv=RAY_memory_monitor_refresh_ms=${RAY_memory_monitor_refresh_ms:-}"
  )
  [ -n "${VIRTUAL_ENV:-}"     ] && env_flags+=("--setenv=VIRTUAL_ENV=${VIRTUAL_ENV}")
  [ -n "${PATH:-}"            ] && env_flags+=("--setenv=PATH=${PATH}")
  [ -n "${PYTHONPATH:-}"      ] && env_flags+=("--setenv=PYTHONPATH=${PYTHONPATH}")
  [ -n "${LD_LIBRARY_PATH:-}" ] && env_flags+=("--setenv=LD_LIBRARY_PATH=${LD_LIBRARY_PATH}")

  case "${HIGH_PRIO_MODE}" in
    rt)
      if command -v chrt >/dev/null 2>&1; then
        if [ "$(id -u)" -eq 0 ] || [ -n "${SUDO:-}" ]; then
          ${SUDO:-} chrt --rr "${RT_PRIO}" ionice -c2 -n0 taskset -c "${ALLOWED_CPUS}" "$@"
          return
        else
          echo "Aviso: modo rt requer root/sudo. Caindo para 'batch'." >&2
        fi
      fi
      ;;&  # se RT não rolou, continua

    service)
      if command -v systemd-run >/dev/null 2>&1; then
        # Service transitório: pode usar --wait
        "${run_sysd[@]}" --quiet --collect --same-dir --wait \
          "${env_flags[@]}" \
          -p CPUWeight="${CPU_WEIGHT}" -p IOWeight="${IO_WEIGHT}" \
          bash -lc 'if nice -n -5 true 2>/dev/null; then exec nice -n -5 ionice -c2 -n0 taskset -c "$ALLOWED_CPUS" "$@"; else exec ionice -c2 -n0 taskset -c "$ALLOWED_CPUS" "$@"; fi' _ "$@"
        return
      fi
      ;;&

    batch|*)
      if command -v systemd-run >/dev/null 2>&1; then
        # Scope: **sem --wait** (incompatível). O comando roda em foreground e o shell segue quando ele termina.
        "${run_sysd[@]}" --quiet --collect --same-dir --scope \
          "${env_flags[@]}" \
          -p CPUWeight="${CPU_WEIGHT}" -p IOWeight="${IO_WEIGHT}" \
          bash -lc 'if nice -n -5 true 2>/dev/null; then exec nice -n -5 ionice -c2 -n0 taskset -c "$ALLOWED_CPUS" "$@"; else exec ionice -c2 -n0 taskset -c "$ALLOWED_CPUS" "$@"; fi' _ "$@"
      else
        # Fallback sem systemd-run
        if nice -n -5 true 2>/dev/null; then
          taskset -c "${ALLOWED_CPUS}" nice -n -5 ionice -c2 -n0 "$@"
        else
          taskset -c "${ALLOWED_CPUS}" ionice -c2 -n0 "$@"
        fi
      fi
      ;;
  esac
}

# ------------------------ Workflow original ------------------------------------
# Goes to python dir
cd "../../"

# Run each configuration 3 times
for i in {1..5}; do
  echo "Seed $i"
  # criar modelo
  echo "Criando modelo"
  python gen_sim_model.py --seed $i

  # criar perfis
  echo "Criando perfis"
  python gen_sim_profile.py --seed $i

  for alpha in 0.1 0.3 1.0; do
    echo "Alpha $alpha"
    flwr run . gpu-sim-lrc --run-config="seed=$i num-rounds=150 participants-name='hetaaff' dir-alpha=$alpha"
  done
done