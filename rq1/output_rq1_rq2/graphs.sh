export PYTHONPATH=$(pwd)/..

cat B1*.metrics > /tmp/metrics

poetry run datacollect seed_metrics /tmp/metrics
poetry run datacollect combined_plots_percent /tmp/metrics
poetry run datacollect valid_files_percentage /tmp/metrics                  
poetry run datacollect seed_args_and_funcs  /tmp/metrics

cat B1*.coverage > /tmp/coverage

poetry run datacollect coverage_means /tmp/coverage /tmp/metrics




