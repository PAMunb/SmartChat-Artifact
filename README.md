# Artifacts for the Paper "*Title Pending*"

This repository contains the experimental artifacts supporting our research.

---

## Setup Instructions

To reproduce the results, follow the steps below:

### 1. Clone the Repository and Submodules

```bash
git clone https://github.com/your-org/SmartChat-Artifact.git
cd SmartChat-Artifact
git submodule update --init --recursive
```

### 2. Install Dependencies

- Install [Poetry](https://python-poetry.org/docs/#installation) for Python package management.
- Install [.NET SDK 5.0](https://dotnet.microsoft.com/en-us/download/dotnet/5.0) to build SmartChat.
- Install [Docker](https://docs.docker.com/get-docker/) if not already available.

### 3. Build and Configure

```bash
cd SmartChat
make
```

Then, install Python dependencies using Poetry:

```bash
cd script
poetry install

cd ../scripts
poetry install
```

---

## RQ1: LLMs for Initial Fuzzing Seed Generation

To generate the graphs for RQ1:

```bash
cd rq1
tar -xJf data.tar.xz
./get_metrics.sh
./get_coverage.sh
./graphs.sh
```

---

## RQ2: Ablation and Temperature Comparison

To generate the results and plots:

```bash
cd rq2
tar -xJf *.tar.xz

./create_data.sh ./output_B1_llm_only_buggain llm_only_buggain
./create_data.sh ./output_B1_llm_only llm_only
```

Generate bar plots:

```bash
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

To extract and prepare data:

```bash
cd rq3
tar -xJf *.tar.xz
./create_data.sh
```

### Plotting for GPT4.1-Mini:

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
  --left-data-files ./smartian.bug ./confuzzius.bug \
    ./smartchat_cot_gpt4.1mini_0.4.bug \
    ./smartchat_code_gpt4.1mini_0.4.bug \
    ./smartchat_nocode_gpt4.1mini_0.4.bug \
  --left-ylabel "Total # of Bugs found" \
  --ylabel "Instruction Coverage (%)"
```

### Plotting for Llama3.3-70B:

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
  --left-data-files ./smartian.bug ./confuzzius.bug \
    ./smartchat_cot_Llama3.3-70B_0.4.bug \
    ./smartchat_code_Llama3.3-70B_0.4.bug \
    ./smartchat_nocode_Llama3.3-70B_0.4.bug \
  --left-ylabel "Total # of Bugs found" \
  --ylabel "Instruction Coverage (%)"
```
