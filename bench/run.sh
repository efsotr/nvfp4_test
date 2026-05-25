python bench_vllm.py > results_vllm.txt
python bench16.py --load default > results_16_default.txt
python bench16.py --load sep > results_16_global.txt
python bench16.py --load trans > results_16_trans.txt