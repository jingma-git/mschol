# Issues: OpenCV conflicit with OpenBLAS 
I've tested run_problem_2d.sh, it works perfectly. But when I try using the ichol solver in my implementation for paper "Scale-Space and Edge Detection Using Anisotropic Diffusion". anidiff did not work, stuck on openblas cholesky solver dpotrf_()  (iteration_body(), ichol.h).

You can reproduce the bug by
```
cd makefiles;
./run_anidiff.sh
```

Then you can go to the "result" directory, the amg solver works OK. You can find an edge-preserved blurred image iter0.png. But ichol just stuck on dpotrf_(), ichol.h.
# Tested Environment
Ubuntu20.04 LTS, OpenCV 4.3.0, CPU: AMD® Ryzen 7 3700x 8-core processor × 16 

# Ad-hoc remedy: separate OpenCV routine from the ichol solver
```
./run_anidiff2.sh
```

# Compare Git commit difference
https://github.com/jingma-git/mschol/compare/5625adb..7be281c
https://docs.github.com/en/github/committing-changes-to-your-project/viewing-and-comparing-commits/comparing-commits

# Impl for edge-preserved smoothing (only depend on OpenCV and Eigen)
https://github.com/jingma-git/anistropic_diffusion/