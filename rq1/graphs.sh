export PYTHONPATH=$(pwd)/../scripts

cat B1*.metrics > ./metrics
cat B1*.coverage > ./coverage

poetry --directory ../scripts run python ../scripts/datacollect.py seed_metrics ./metrics
poetry --directory ../scripts run python ../scripts/datacollect.py combined_plots_percent ./metrics
poetry --directory ../scripts run python ../scripts/datacollect.py valid_files_percentage ./metrics
poetry --directory ../scripts run python ../scripts/datacollect.py seed_args_and_funcs ./metrics
poetry --directory ../scripts run python ../scripts/datacollect.py coverage_means ./coverage --metrics_csv ./metrics 

