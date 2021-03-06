#!/bin/bash

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)/2" | bc`
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

max_e=1e5
min_e=1e1
monthday=$(date +"%Y-%m-%d-%H-%M-%S")/patt-${max_e}-${min_e}/

for precond in amg ichol; do # amg
for prb_name in laplacian; do # [laplacian | elasticity]
for strategy in regular; do # [regular | random]
for num_rand_samples in 9; do # 9 90601:301x301 200000 500000 800000
for mtr_name in RAND08; do
for rho in 07.0; do # 07.5 08.0
    if [ "$precond" != "ichol" ]; then
	rho=00
    fi
    new_num_samples=${num_rand_samples}
    if [ "$prb_name" == "elasticity" ]; then
	new_num_samples=`echo "${num_rand_samples}/2" | bc`
    fi
    date=${monthday}/prb-${prb_name}/prec-${precond}
    make -f run_chol_odw.mk anidiff DATE=${date} NEI_NUM=${rho} \
	 MTR_NAME=${mtr_name} MAX_E=${max_e} MIN_E=${min_e} \
	 NUM_THREADS=${num_threads} PRECOND=${precond} \
	 SUBST_NUM_THREADS=${num_threads} NUM_RAND_SAMPLES=${new_num_samples} \
	 SAMPLE_STRATEGY=${strategy} PRB_NAME=${prb_name} \
	 COARSEST_NODE_NUM=25 IMG_H=20 IMG_W=32
done
done
done
done
done
done
