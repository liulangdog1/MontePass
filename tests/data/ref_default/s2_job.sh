#!/bin/bash
#SBATCH --job-name=q_min_it1_centre_s2
#SBATCH --chdir=@ROOT@/min/it1/centre_s2
#SBATCH --output=@ROOT@/min/it1/centre_s2/job.out
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=testaccount
#SBATCH --qos=testqos
#SBATCH --partition=testpartition
#SBATCH --constraint=testfeature
#SBATCH --no-requeue
set -u
cd @ROOT@/min/it1/centre_s2 || exit 97
module load qmc-tools || { echo SETUP_FAILED: module load qmc-tools; exit 96; }
export PATH=/opt/qmcpack/bin:$PATH || { echo SETUP_FAILED: export PATH=/opt/qmcpack/bin:$PATH; exit 96; }
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/opt/qmcpack/src/QMCTools:${PYTHONPATH:-}
export PYTHONPATH=@PKG@:${PYTHONPATH:-}

# ---------------------------------------------------------------------------------------------
# Steps of this point. A step is skipped when <step>.done exists. A step writes <step>.started,
# runs under a time limit, checks the exit code and the expected output, and then writes
# <step>.done or <step>.failed (one line that says why).
# ---------------------------------------------------------------------------------------------
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }
mark() { echo "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"; }
begin_step() {
    STEP="$1"
    if [ -e "$STEP.done" ]; then say "step $STEP: done before, skipped"; return 1; fi
    if [ -e stop.requested ]; then say "step $STEP: not started, a stop was requested"; exit 0; fi
    rm -f "$STEP.failed"
    mark "$STEP.started" "$(date +%s)"
    say "step $STEP: started"
    return 0
}
end_step() { mark "$STEP.done" "$(date +%s)"; say "step $STEP: done"; }
fail_step() {
    mark "$STEP.failed" "$*"
    say "step $STEP: FAILED: $*"
    exit 1
}
# run_limited SECONDS OUTPUT ERRORS COMMAND ...   (ERRORS "-" = into OUTPUT). The exit code is left in RC.
run_limited() {
    LIMIT="$1"; run_out="$2"; run_err="$3"; shift 3
    WHAT="$*"
    if [ "$run_err" = "-" ]; then
        run_err="$run_out"
        timeout -k 30 "$LIMIT" "$@" > "$run_out" 2>&1
        RC=$?
    else
        timeout -k 30 "$LIMIT" "$@" > "$run_out" 2> "$run_err"
        RC=$?
    fi
}
# check_run: the step fails unless the last command ended with exit code 0
check_run() {
    if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then fail_step "TIMEOUT: no end within $LIMIT s: $WHAT"; fi
    if [ "$RC" -ne 0 ]; then
        hint=""
        if grep -q -E "UCX|MPI_INIT" "$run_err" 2>/dev/null; then hint=" [MPI start failure, see $run_err]"; fi
        fail_step "RUN: exit code $RC: $WHAT$hint"
    fi
}
need_text() { grep -q -- "$2" "$1" 2>/dev/null || fail_step "RUN: the text '$2' is not in $1"; }
need_file() { [ -s "$1" ] || fail_step "RUN: the file $1 is missing or empty"; }
# package COMMAND: a command of the package. Exit code 3 = a quality check failed; the last line of its output says why.
package() {
    run_limited 900 "$1.out" - python3 -m montepass.engine.qmcpack "$1"
    if [ "$RC" -eq 3 ]; then fail_step "QUALITY: $(tail -n 1 "$1.out")"; fi
    check_run
}
count_blocks() { awk '!/^#/ && NF > 0 { n++ } END { print n + 0 }' "$1" 2>/dev/null; }
say "job ${SLURM_JOB_ID:-local} on $(hostname), nodes ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "$(date +%s) job=${SLURM_JOB_ID:-local} nodes=${SLURM_JOB_NODELIST:-$(hostname)}" >> job.info

if begin_step dmc; then
    rm -f dmc2.s000.scalar.dat dmc2.s001.scalar.dat      # blocks of an earlier attempt are not blocks of this run
    need_file dmc2.in.xml
    run_limited 7200 dmc2.out dmc2.err mpirun -np 32 --map-by core --bind-to core qmcpack_real dmc2.in.xml
    blocks=$(count_blocks dmc2.s001.scalar.dat); blocks=${blocks:-0}
    if [ -e stop.requested ]; then
        mark dmc.stopped "$blocks"
        say "step dmc: stopped on request after $blocks blocks"
        exit 0
    fi
    check_run
    need_text dmc2.out "QMCPACK execution completed successfully"
    if [ "$blocks" -lt 400 ]; then fail_step "RUN: dmc2.s001.scalar.dat holds $blocks of 400 blocks"; fi
    end_step
fi
say "all steps of this job are done"
exit 0
