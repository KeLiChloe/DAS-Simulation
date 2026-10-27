Experiment setup snapshot
=========================
source_script : /home/5880.li/DAS-Simulation/run_scripts/discrete/run_K.sh
copied_as     : exp_oct_2026/discrete/varying_K/set_6/run_script.sh
saved_utc     : 2026-10-27T21:38:31Z
host          : gpu-1-cii-01

How to re-read parameters
-------------------------
1. Open run_script.sh in this folder (exact bash used for the sweep).
2. Each *.pkl stores main.py "exp_params" (K/d/noise ranges, algorithms, cv_folds, ...).

Re-run tip: bump SAVE_DIR (e.g. set_3) before launching so you do not overwrite.
