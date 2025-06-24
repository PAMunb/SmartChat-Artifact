export PYTHONPATH=$(pwd)/../scripts

cat B1*.metrics > ./metrics
cat B1*.coverage > ./coverage

python3.8 ../scripts/datacollect.py seed_metrics ./metrics
python3.8 ../scripts/datacollect.py combined_plots_percent ./metrics
python3.8 ../scripts/datacollect.py valid_files_percentage ./metrics
python3.8 ../scripts/datacollect.py seed_args_and_funcs ./metrics
python3.8 ../scripts/datacollect.py coverage_means ./coverage --metrics_csv ./metrics 

