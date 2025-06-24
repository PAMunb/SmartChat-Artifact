# Artifacts for the Paper "*Title Pending*"

This repository contains the experimental artifacts for the paper.

## Setup

Clone this repository and initialize the submodules:

```bash
git clone https://github.com/<your-org>/SmartChat-Artifact.git
cd SmartChat-Artifact
git submodule update --init --recursive
```

Then, install the following dependencies:

1. [Python 3.8](https://www.python.org/downloads/release/python-380/)
2. [Poetry](https://python-poetry.org/docs/#installation)
3. [.NET 5.0 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/5.0) (required to run SmartChat)

Build SmartChat:

```bash
cd SmartChat
make
```

---

## RQ1: LLMs for Initial Fuzzing Seed Generation

To generate the graphs for this research question, run:

```bash
cd rq1
tar -xJf data.tar.xz
./get_metrics.sh
./get_coverage.sh
./graphs.sh
```

---

## RQ2: Ablation and Temperature Comparison

To generate the graphs for this research question:

```bash
cd rq2
tar -xJf *.tar.xz

./create_data.sh ./output_B1_llm_only_buggain llm_only_buggain
./create_data.sh ./output_B1_llm_only llm_only

python3.8 ../scripts/plot_b1_bar.py ./llm_only_buggain/merged_all_llm_only_buggain* \
  --models "GPT4.1-Mini,Llama3.3-70B" \
  --colors "#8facd2,#dd7e7e" \
  --output llm_only_buggain.pdf

python3.8 ../scripts/plot_b1_bar.py ./llm_only/merged_all_llm_only* \
  --models "GPT4.1-Mini,Llama3.3-70B" \
  --colors "#8facd2,#dd7e7e" \
  --output llm_only.pdf
```

### Statistical Evaluation

```bash
python3.8 ../scripts/statistical_eval.py ./llm_only_buggain/llm_only_buggain.csv base_dfa.csv
python3.8 ../scripts/statistical_eval.py ./llm_only_buggain/llm_only_buggain.csv base_rand.csv
```

---

## RQ3: Impact of Source Code Inclusion

To generate the graphs for this research question:

```bash
cd rq3
tar -xJf *.tar.xz
./create_data.sh
```

### Using GPT4.1-Mini:

```bash
python3.8 ../../scripts/plot_b2_graph.py \
  ./smartian.cov ./confuzzius.cov \
  ./smartchat_cot_gpt4.1mini_0.4.cov \
  ./smartchat_code_gpt4.1mini_0.4.cov \
  ./smartchat_nocode_gpt4.1mini_0.4.cov \
  -o comparison_plot_combined_gpt.pdf \
  --baseline-name "Smartian" \
  --method-names "Confuzzius" "SmartChat CoT Code (gpt4.1mini)" "SmartChat Code (gpt4.1mini)" "SmartChat ABI (gpt4.1mini)" \
  --left-ylim 60 95 \
  --legend-loc "best" \
  --subplot-layout \
  --left-data-files ./smartian.bug ./confuzzius.bug ./smartchat_cot_gpt4.1mini_0.4.bug \
                     ./smartchat_code_gpt4.1mini_0.4.bug ./smartchat_nocode_gpt4.1mini_0.4.bug \
  --left-ylabel "Total # of Bugs found" \
  --ylabel "Instruction Coverage (%)"
```

### Using Llama3.3-70B:

```bash
python3.8 ../../scripts/plot_b2_graph.py \
  ./smartian.cov ./confuzzius.cov \
  ./smartchat_cot_Llama3.3-70B_0.4.cov \
  ./smartchat_code_Llama3.3-70B_0.4.cov \
  ./smartchat_nocode_Llama3.3-70B_0.4.cov \
  -o comparison_plot_combined_llama.pdf \
  --baseline-name "Smartian" \
  --method-names "Confuzzius" "SmartChat CoT Code (Llama3.3-70B)" "SmartChat Code (Llama3.3-70B)" "SmartChat ABI (Llama3.3-70B)" \
  --left-ylim 60 95 \
  --legend-loc "best" \
  --subplot-layout \
  --left-data-files ./smartian.bug ./confuzzius.bug ./smartchat_cot_Llama3.3-70B_0.4.bug \
                     ./smartchat_code_Llama3.3-70B_0.4.bug ./smartchat_nocode_Llama3.3-70B_0.4.bug \
  --left-ylabel "Total # of Bugs found" \
  --ylabel "Instruction Coverage (%)"
```
