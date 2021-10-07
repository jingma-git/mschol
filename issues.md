I've tested run_problem_2d.sh, it works perfectly. But when I try using the ichol solver in my implementation for paper "Scale-Space and Edge Detection Using Anisotropic Diffusion". anidiff did not work, stuck on openblas cholesky solver dpotrf_()  (iteration_body(), ichol.h).

You can reproduce the bug by
```
cd makefiles;
./run_anidiff.sh
```

Then you can go to the "result" directory, the amg solver works OK. You can find an edge-preserved blurred image iter0.png. But ichol just stuck on dpotrf_(), ichol.h.

The major changes I've made in ichol.h for debug purpose are:
1. line 13
2. line 466, comment the "#pragma omp parallel for" (factorize())
3. line 599, line 631-632, line 686, line 692, line 694-695 (iteration_body())

Other changes I've made: https://github.com/jingma-git/mschol/compare/5625adb..7be281c

Other useful links may help you solve this "bug":
1. https://github.com/xianyi/OpenBLAS/issues/240
2. https://github.com/jingma-git/anistropic_diffusion/
3. https://docs.github.com/en/github/committing-changes-to-your-project/viewing-and-comparing-commits/comparing-commits

I did not gitignore the "result" directory, hopefully, the log file can help you solve the issue.
My OS is Ubuntu20.04 LTS