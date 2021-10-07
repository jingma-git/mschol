EXEC_TRI_LAP = ../build/examples/solve_problem_2d
EXEC_TET_LAP = ../build/examples/solve_problem_3d

DATE ?= 
COARSEST_NODE_NUM ?= 
NEI_NUM ?= 
MTR_NAME ?=
MAX_E ?= 
MIN_E ?=
NUM_THREADS ?= 
SUBST_NUM_THREADS ?= 
NUM_RAND_SAMPLES ?=
SAMPLE_STRATEGY ?= 
PRB_NAME ?= 
ALPHA ?= 1e-4
MAX_SU_SIZE ?= 64

TRI_LAP_OUTDIR = ../result/ichol-$(DATE)/$(MTR_NAME)/tris-$(SAMPLE_STRATEGY)-$(NUM_RAND_SAMPLES)/rho-$(NEI_NUM)/
TRI_LAP_OUT = $(TRI_LAP_OUTDIR)/result.vtk

TET_LAP_OUTDIR = ../result/ichol-$(DATE)/$(MTR_NAME)/tets-$(SAMPLE_STRATEGY)-$(NUM_RAND_SAMPLES)/rho-$(NEI_NUM)/
TET_LAP_OUT = $(TET_LAP_OUTDIR)/result.vtk

trilap: $(TRI_LAP_OUT)
tetlap: $(TET_LAP_OUT)

$(TRI_LAP_OUT): $(EXEC_TRI_LAP)
	@mkdir -p $(TRI_LAP_OUTDIR)
	$(EXEC_TRI_LAP) num_samples=$(NUM_RAND_SAMPLES) wp=1e6 outdir=$(TRI_LAP_OUTDIR) nei_num=$(NEI_NUM) mtr_name=$(MTR_NAME) max_E=$(MAX_E) min_E=$(MIN_E) num_threads=$(NUM_THREADS) subst_num_threads=$(SUBST_NUM_THREADS) precond=$(PRECOND) max_su_size=$(MAX_SU_SIZE) sample_strategy=$(SAMPLE_STRATEGY) prb_name=$(PRB_NAME) alpha=$(ALPHA) coarsest_node_num=$(COARSEST_NODE_NUM) 2>&1 | tee $(TRI_LAP_OUTDIR)/log.txt

$(TET_LAP_OUT): $(EXEC_TET_LAP)
	@mkdir -p $(TET_LAP_OUTDIR)
	$(EXEC_TET_LAP) num_samples=$(NUM_RAND_SAMPLES) wp=1e6 outdir=$(TET_LAP_OUTDIR) nei_num=$(NEI_NUM) mtr_name=$(MTR_NAME) max_E=$(MAX_E) min_E=$(MIN_E) num_threads=$(NUM_THREADS) subst_num_threads=$(SUBST_NUM_THREADS) precond=$(PRECOND) max_su_size=$(MAX_SU_SIZE) sample_strategy=$(SAMPLE_STRATEGY) prb_name=$(PRB_NAME) alpha=$(ALPHA) coarsest_node_num=$(COARSEST_NODE_NUM) 2>&1 | tee $(TET_LAP_OUTDIR)/log.txt
