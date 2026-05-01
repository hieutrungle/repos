# python scripts/run_all_methods_ap_sweep.py --methods all --config configs/run_experiments_cuda_hrbb.json --ap_min 1 --ap_max 6 --seeds 41 42 43 44 --output_dir "tmp_comparison_results1/hrbb_experiments"
# python scripts/run_all_methods_ap_sweep.py --methods all --config configs/run_experiments_cuda_office.json --ap_min 1 --ap_max 6 --seeds 41 42 43 44 --output_dir "tmp_comparison_results1/office_experiments"

python scripts/run_all_methods_ap_sweep.py --methods all --config configs/run_experiments_cuda_office1.json --ap_min 1 --ap_max 6 --seeds 41 42 43 44 --output_dir "tmp_reflector/office_experiments1"
python scripts/run_all_methods_ap_sweep.py --methods all --config configs/run_experiments_cuda_office2.json --ap_min 1 --ap_max 6 --seeds 41 42 43 44 --output_dir "tmp_reflector/office_experiments2"
python scripts/run_all_methods_ap_sweep.py --methods all --config configs/run_experiments_cuda_office3.json --ap_min 1 --ap_max 6 --seeds 41 42 43 44 --output_dir "tmp_reflector/office_experiments3"
python scripts/run_all_methods_ap_sweep.py --methods all --config configs/run_experiments_cuda_office4.json --ap_min 1 --ap_max 6 --seeds 41 42 43 44 --output_dir "tmp_reflector/office_experiments4"
