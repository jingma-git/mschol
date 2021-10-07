#!/bin/bash

num_threads=`echo "$(cat /proc/cpuinfo | grep processor | wc -l)/2" | bc`
export OMP_NUM_THREADS=${num_threads}
echo "===== threads used=${num_threads} ====="

max_e=1e5
min_e=1e1
monthday=$(date -I)/patt-${max_e}-${min_e}/

for precond in ichol amg; do
for prb_name in elasticity; do # [laplacian | elasticity]
for strategy in random; do # [regular | random]
for num_rand_samples in 200000; do # 500000 800000
for mtr_name in RAND64; do
for rho in 03.2; do # 03.2 03.5 03.8 04.0
    if [ "$precond" != "ichol" ]; then
	rho=00
    fi
    new_num_samples=${num_rand_samples}
    if [ "$prb_name" == "elasticity" ]; then
	new_num_samples=`echo "${num_rand_samples}/3" | bc`
    fi
    date=${monthday}/prb-${prb_name}/prec-${precond}
    make -f run_chol_odw.mk tetlap DATE=${date} NEI_NUM=${rho} \
	 MTR_NAME=${mtr_name} MAX_E=${max_e} MIN_E=${min_e} \
	 NUM_THREADS=${num_threads} PRECOND=${precond} \
	 SUBST_NUM_THREADS=${num_threads} NUM_RAND_SAMPLES=${new_num_samples} \
	 SAMPLE_STRATEGY=${strategy} PRB_NAME=${prb_name} \
	 COARSEST_NODE_NUM=125
done
done
done
done
done
done
