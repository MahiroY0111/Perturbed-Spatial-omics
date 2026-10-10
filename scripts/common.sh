# Shared settings of the job scripts (sourced, not run). Set TH (threads per container) before sourcing.
#   PYSVC  container for training and prediction: py (CPU, default) or pygpu (NVIDIA GPU)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE="$REPO/environment/compose.yaml"
TH=${TH:-4}
PYSVC=${PYSVC:-py}
ENVS=(-e OMP_WAIT_POLICY=passive -e GOMP_SPINCOUNT=0 -e KMP_BLOCKTIME=0 -e OMP_NUM_THREADS=$TH -e OPENBLAS_NUM_THREADS=$TH
      -e MKL_NUM_THREADS=$TH -e NUMBA_NUM_THREADS=$TH -e TQDM_DISABLE=1)
dc() { local svc=$1; shift; docker compose -f "$COMPOSE" run --rm -T "${ENVS[@]}" "$svc" "$@"; }
have() { [ -e "$REPO/$1" ]; }
