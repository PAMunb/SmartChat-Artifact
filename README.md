# Artifacts of the paper "."

This repository contains artifacts for the experiments in the paper...

1. install Poetry
2. 

## RQ1: 

To generate the graphs of this reseach question, follow the instructions:

```
$ cd rq1

$ tar -xJf data.tar.xz

$ ./get_metrics.sh

$ ./get_coverage.sh

$ ./graphs.sh

```

## RQ2:

To generate the graphs of this reseach question, follow the instructions:

```
$ cd rq2

$ tar -xJf *.tar.xz

$ ./create_data.sh ./output_B1_llm_only_buggain llm_only_buggain

$ ./create_data.sh ./output_B1_llm_only llm_only

$ python3.8 ../scripts/plot_b1_bar.py ./llm_only_buggain/merged_all_llm_only_buggain* --models "GPT4.1-Mini,Llama3.3-70B"  --colors "#8facd2,#dd7e7e" --output llm_only_buggain.pdf

$ python3.8 ../scripts/plot_b1_bar.py ./llm_only/merged_all_llm_only* --models "GPT4.1-Mini,Llama3.3-70B"  --colors "#8facd2,#dd7e7e" --output llm_only.pdf

```
Regarding the statistical evaluation, proced as bellow:

```
```

## RQ3: 

To generate the graphs of this reseach question, follow the instructions:

```
$ cd rq3

$ tar -xJf *.tar.xz

$ ./create_data.sh

$ python3.8 ../../scripts/plot_b2_graph.py   ./smartian.cov ./confuzzius.cov ./smartchat_cot_gpt4.1mini_0.4.cov ./smartchat_code_gpt4.1mini_0.4.cov ./smartchat_nocode_gpt4.1mini_0.4.cov   -o comparison_plot_combined_gpt.pdf   --baseline-name "Smartian"   --method-names "Confuzzius" "SmartChat CoT Code (gpt4.1mini)" "SmartChat Code (gpt4.1mini)" "SmartChat ABI (gpt4.1mini)"  --left-ylim  60 95    --legend-loc "best"   --subplot-layout   --left-data-files ./smartian.bug ./confuzzius.bug ./smartchat_cot_gpt4.1mini_0.4.bug ./smartchat_code_gpt4.1mini_0.4.bug ./smartchat_nocode_gpt4.1mini_0.4.bug   --left-ylabel "Total # of Bugs found"   --ylabel "Instruction Coverage (%)"

$ python3.8 ../../scripts/plot_b2_graph.py   ./smartian.cov ./confuzzius.cov ./smartchat_cot_Llama3.3-70B_0.4.cov ./smartchat_code_Llama3.3-70B_0.4.cov ./smartchat_nocode_Llama3.3-70B_0.4.cov   -o comparison_plot_combined_llama.pdf   --baseline-name "Smartian"   --method-names "Confuzzius" "SmartChat CoT Code (Llama3.3-70B)" "SmartChat Code (Llama3.3-70B)" "SmartChat ABI (Llama3.3-70B)"  --left-ylim  60 95    --legend-loc "best"   --subplot-layout   --left-data-files ./smartian.bug ./confuzzius.bug ./smartchat_cot_Llama3.3-70B_0.4.bug ./smartchat_code_Llama3.3-70B_0.4.bug ./smartchat_nocode_Llama3.3-70B_0.4.bug   --left-ylabel "Total # of Bugs found"   --ylabel "Instruction Coverage (%)"

```
