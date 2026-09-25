# E3 — generalization to post-cutoff contracts: selection protocol

This page documents in full the construction of the contract sample used to replicate the
SmartChat vs. Smartian comparison on code that postdates the models, and the campaign protocol
applied to it. The paper states the criteria and the reasoning in its threats-to-validity
section; what follows is the operational detail, including the numbers that a reader would need
to judge how selective the sample is, and the commands to reproduce it.

Scripts: `select-recent-contracts.py` (sampling and source-level filters) and
`gate-recent-contracts.sh` (the execution gate), both in the paper's revision directory.

## Why a new sample at all

Two threats motivate it, and it addresses them to different degrees.

**Generalization.** The published results come from Bench58 and Bench78. Both are curated sets
of older contracts, so the obvious question is whether LLM-generated seeds remain competitive
outside them.

**Contamination.** Every contract in those benchmarks predates the training cutoff of every
model we evaluate, so memorization is a live alternative explanation for our numbers.

The sample bounds verbatim memorization, since none of these contracts existed when the models
were trained. It does **not** establish unfamiliarity: contracts that execute in isolation,
which is what a fuzzer needs, tend to derive from widely used templates, and a model can
recognize a pattern it has seen many times without having seen this deployment. The stronger
evidence on contamination comes from the identifier-renaming experiment (E1), which measures
the *effect* of contamination on the benchmark the paper actually reports.

## No ground truth is required

The comparison judges both configurations — SmartChat's LLM seeds and Smartian's data-flow
seeds — with the same runtime oracles, on the same contracts, under the same budget. Whether a
given alarm is a true positive is unknown and does not need to be known: false alarms affect
both sides equally, so the comparison remains fair. The claim the experiment supports is
therefore "LLM seeds drive the fuzzer to as many alarms, as quickly, as data-flow seeds do on
contracts the models cannot have seen", not "LLM seeds find N real bugs".

## Where candidates come from

From the index of source-verified contracts (Sourcify), **not** from scanning block contents.
Contemporary deployments are performed almost entirely by factory contracts, which leave no
top-level creation transaction: six randomly chosen recent mainnet blocks contained 1,353
transactions and not a single creation. A block scan therefore finds almost nothing, and what
it does find is unrepresentative.

Two consequences of this choice are worth stating:

- **Eligibility uses the deployment block, not the verification date.** The index is
  continuously fed with contracts of every age — the first candidate examined during
  development was deployed in 2017 and verified the same week. The cut-off date is translated
  once into a block number (`getblocknobytime`) and each candidate is a plain integer
  comparison against it.
- **Nothing is compiled locally.** The index supplies the ABI and the on-chain creation
  bytecode, which is what the fuzzer consumes. This removes the need to pin solc versions or to
  choose an `--evm-version`, and with it a whole class of mismatch between our binaries and the
  deployed ones. A contract compiled for a fork newer than Istanbul simply fails to deploy
  under Smartian's embedded EVM, and the execution gate below catches it.

## Source-level criteria

A candidate is retained only if all hold:

| # | Criterion | Why |
|---|---|---|
| i | deployed after the latest training cutoff among the evaluated models | the point of the sample |
| ii | not a proxy (the index's own proxy resolution, plus a `delegatecall`/proxy-pattern check on the file declaring the contract) | a proxy runs code that lives elsewhere, so fuzzing it measures the wrong thing |
| iii | constructor takes no arguments | the deployment transaction has to succeed unattended |
| iv | at least two state-changing public or external functions with arguments | otherwise there is no behaviour for a seed to explore |
| v | at most two hardcoded mainnet addresses in that file | interesting paths would revert in an offline EVM |
| vi | creation bytecode present and within the EIP-170 limit of 24 KB | it must be deployable |

Only the file declaring the contract under test is inspected for (ii) and (v): a library it
imports may mention `delegatecall` without the contract being a proxy.

**Deduplication.** Recent deployments are dominated by boilerplate — token clones above all.
Candidates with identical creation bytecode are dropped, and clones that differ only in
embedded constants (a token's name and symbol, for instance) are additionally grouped by
bytecode *structure*, with at most one contract drawn per group. Without this the sample would
measure a handful of templates many times over and overstate its own diversity. The grouping
reuses the equivalence check built for E1: basic blocks compared with immediates masked.

## Execution gate

Source-level filters cannot tell whether a contract does anything when executed offline. Each
surviving candidate is therefore run through a short campaign and kept only if it **deploys**
and reaches **non-trivial instruction coverage**. This is the filter that ultimately defines
the sample, and the one most likely to reject: a contract whose every entry point reverts
without external state contributes nothing but noise to either side of the comparison.

## Draw

The final contracts are drawn at random, with a fixed seed recorded in the manifest, from the
candidates that pass the gate, at most one per structural group. The manifest lists for each
selected contract its address, name, compiler version, deployment block and bytecode digest,
so the sample can be reconstructed exactly.

## Campaign protocol

As in the rest of the paper: 1-hour campaigns, five repetitions, one dedicated physical core
per trial, in a container pinned to that core. Three configurations — SmartChat with
GPT-4.1-mini seeds, SmartChat with Llama3.3-70B seeds, and Smartian with its own data-flow
seeds. Seed generation follows RQ2: the ABI-only prompt, temperature 0.4, ten test cases of at
least four transactions, with an independent seed set per repetition.

Metrics: alarms raised per oracle, time to the first alarm, and instruction coverage, compared
across configurations with the Mann-Whitney U test and the Vargha-Delaney effect size. Effect
sizes are reported alongside p-values because with five repetitions a test of this kind can
only rule out large differences, so a p-value above 0.05 is absence of evidence of a
difference and never evidence of equivalence.

## Funnel

<!-- to fill from e3-candidates/manifest.json once the gate has run -->

| Stage | Contracts |
|---|---|
| verified contracts examined | |
| rejected: deployed before the cutoff | |
| rejected: proxy | |
| rejected: constructor takes arguments | |
| rejected: fewer than two state-changing functions with arguments | |
| rejected: hardcoded addresses | |
| rejected: duplicate bytecode | |
| candidates passing the source-level filters | |
| rejected by the execution gate | |
| **drawn for the experiment** | **40** |

## Reproducing it

    ETHERSCAN_API_KEY=... ./select-recent-contracts.py --out e3-candidates --target 120
    ./gate-recent-contracts.sh e3-candidates            # deploy + coverage, on the cluster
    ./run-campaigns.py --arms e3-sample --seeds e3-seeds --out e3-campaigns \
        --cores 0-31 --mems 0

The Etherscan key is needed only to turn the cut-off date into a block number; everything else
comes from the open verified-contract index.
