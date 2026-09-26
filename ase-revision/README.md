# E1 — identifier renaming on Bench58 (ASE/EMSE revision round)

Experiment E1 asks whether the performance reported in the paper is inflated by the models
having seen the benchmark contracts during training. It does not try to prove that the
contracts were absent from the training data, which is unanswerable for closed-weight models;
it bounds the *effect* of contamination, which is measurable.

Bench58 is fuzzed in three arms that differ only in their identifiers — the bytecode is
equivalent, and the fuzzer never reads identifiers, only the LLM that generates the seeds
does:

| Arm | Transformation | Removes |
|---|---|---|
| A | original contract | — |
| B | synonym renaming (`transfer` -> `sendTokens`, `BecToken` -> `AuroraAsset`) | exact-string memorization; meaning preserved |
| C | opaque renaming (`f1`, `v1`) | memorization **and** meaning |

Arm C is what makes arm B interpretable: `transfer(address,uint256)` tells a model what the
function does, and that is domain knowledge rather than memorization.

## Contents

| Archive | Unpacked | What it holds |
|---|---|---|
| `e1-batch2-campaigns.tar.xz` | 560 MB | the other 1,740 campaigns: the same arms with Llama3.3-70B seeds, and Smartian's own data-flow seeds as the validity control |
| `e1-trials.csv`, `e1-trial-status.txt` | — | the state of every one of the 2,610 trials, and the exclusion list derived from it |
| `e1-analysis-inputs.tar.xz` | — | the CVE addresses mapped onto the renamed arms, the campaign logs and the scripts |
| `e1-batch1-campaigns-gpt4.1mini.tar.xz` | 281 MB | 870 one-hour campaigns: 3 arms x 58 contracts x 5 repetitions, with SmartChat seeds from GPT-4.1-mini. Per trial: `log.txt`, `cov.txt`, `testcase/`, `bug/`, `with_dfeed.txt`, `without_dfeed.txt` |
| `e1-seeds.tar.xz` | 288 MB | the 30 seed sets (17,334 seeds) and the raw model responses they came from |
| `e1-corpus-logs-scripts.tar.xz` | 14 MB | the three-arm corpus (`.sol`, `.abi`, `.bin` per contract per arm), the campaign and generation logs, and the scripts |

`SHA256SUMS` covers all three.

## How it was produced

Solc 0.4.26 with no optimizer reproduces all 58 shipped binaries byte-for-byte apart from the
trailing metadata hash, so **arm A is the published benchmark itself** and serves as a sanity
check on the pipeline. Campaigns follow the paper's protocol: one-hour budget, five
repetitions, one dedicated physical core per trial (`--cores 0-31 --mems 0` on a two-socket
Xeon Gold 6338, one socket left idle so trials do not contend for memory bandwidth — time to
detection is one of the metrics).

Repetition *i* consumes seed set *i*. The paper reuses a single seed set across its five
repetitions, so those capture only the fuzzer's stochasticity; pairing them makes the
repetitions capture the generator's as well, at the cost of extra API calls only.

## Reading the numbers

The scripts in the third archive regenerate the analysis:

    ./classify-trials.py --campaigns <dir>     # per-trial state, and the exclusion list
    ./analyze-e1.py --campaigns <dir> --bug-csv benchmarks/assets/B1-bug.csv

Detection uses the artifact's own criterion (`IntegerBug at` in `log.txt`), as does the time
to detection, so these numbers read against the published ones.

Five contracts are excluded from **all** arms because at least one of their trials did not
complete: `2018-10376` (the fuzzer is killed at ~18 min under the 6 GB cap in arms B and C),
and `2018-13202`, `2018-13220`, `2018-13625`, `2018-14063` (unhandled exceptions in Smartian's
seed handling, all in arm C; the last of these has an empty seed set, which crashes the
fuzzer outright). Dropping them from every arm keeps the comparison symmetric — keeping them
would manufacture a difference caused by tool limits rather than by memorization.

## Result, batch 1 (GPT-4.1-mini seeds, 5 repetitions, 53 contracts)

| Arm | Detected | Time to detection | Instruction coverage |
|---|---|---|---|
| A | 48.8 | 149 s | 3024 |
| B | 48.0 | 153 s | 3019 |
| C | 48.2 | 218 s | 3020 |

No significant difference anywhere (Mann-Whitney, repetition as the unit). Detection and
coverage are flat across arms; arm C is about 46% slower to the first alarm than arm A
(A12 = 0.16, p = 0.095), while arms A and B are indistinguishable in timing (A12 = 0.48).

With five repetitions a test of this kind can only rule out large differences, so these
p-values are reported alongside effect sizes and must not be read as proof of equivalence.

Full results, including the Llama3.3-70B arms and the Smartian validity control, are in
[E1-protocol.md](E1-protocol.md), together with the renamer, the equivalence check, the
detection criterion and how it was carried across the arms, the excluded contracts, and the
statistics. In short: detection is 48.8/47.6/47.4 for GPT-4.1-mini and 48.8/47.8/47.4 for
Llama3.3-70B across arms A, B and C, and Smartian's identifier-blind data-flow mode is flat
(48.0/48.2/47.6) — which is what licenses reading the other rows.
