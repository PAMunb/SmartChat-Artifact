# E1 — identifier renaming on Bench58: protocol and full results

The manuscript's threats-to-validity section states the design and the outcome. This page holds
what does not belong in the body: how the arms were built and verified, which contracts were
excluded and why, how the detection criterion was carried across the arms, and the complete
statistics.

Scripts (in the paper repository, `ase_review_comments/`): `renamer/renamer.js`,
`renamer/buildarms.js`, `renamer/mapcve.js`, `generate-seeds.sh`, `run-campaigns.py`,
`classify-trials.py`, `analyze-e1.py`. Data: the archives beside this file.

## The question the experiment answers

Not "were these contracts in the training data?", which is unanswerable for closed-weight
models and which any design claiming to answer it will be attacked over, but "is the reported
performance inflated by memorization?" The first is about exposure, the second about effect,
and only the second threatens the paper's conclusions.

## The three arms

| Arm | Transformation | What it removes |
|---|---|---|
| A | original contract | — |
| B | identifiers replaced by synonyms (`transfer` -> `sendTokens`, `balances` -> `accountBalances`, `BecToken` -> `AuroraAsset`) | exact-string memorization; meaning preserved |
| C | identifiers replaced by opaque names (`f1`, `v1`) | memorization **and** meaning |

Arm C is what makes arm B interpretable. `transfer(address,uint256)` tells a model what the
function does, and that is domain knowledge rather than recall; opaque renaming destroys both at
once, so a drop in B alone cannot be attributed to either cause without C.

Renaming is done on the compiler's AST (solc-js 0.4.26; all Bench58 contracts are Solidity
0.4.x), not on a separate analysis framework: solc's compact AST carries
`referencedDeclaration` on every reference, which is exactly the property that makes renaming
safe, and it avoids a second toolchain.

Decisions forced by the implementation, all of which affect how the arms should be read:

- **Comments are stripped from all three arms, including A**, so that the arms differ only in
  identifiers. NatSpec binds parameter names, so renaming a parameter without editing its
  docstring is a compile error. Comments do not affect bytecode, and this matches the
  preprocessing the paper already applies to source-code prompts.
- **Contract names are renamed together with their constructors.** A contract name identifies a
  deployment — `BecToken` is the contract behind CVE-2018-10299 — and is therefore the strongest
  memorization cue in the source. Proper-noun segments are replaced from a fixed list of neutral
  stems while generic segments are kept, so arm B preserves *what kind of contract this is* and
  destroys *which one*: `SafeMath` becomes `GuardedMath`, `ERC20Interface` is left alone.
- Identifiers inside inline assembly (7 of them), struct members, enum values and events are
  left alone, each for a reason documented in the renamer's README.
- **Arm B coverage:** 90.9% of ABI-visible identifiers receive a genuine synonym; 6.9% fall back
  to an `alt` prefix that retains the original token, weakening de-memorization for those; 2.1%
  are generic names such as `ERC20` and `Ownable`, kept deliberately.
- **Residual limitation:** string literals are not rewritten, so a contract setting
  `name = "Dimon Coin"` still carries its token name. Rewriting literals would change the
  bytecode and force the equivalence check to exempt string pushes; we keep the check strict and
  declare the residue.

## Equivalence check

Renaming changes selectors, and solc 0.4 orders the dispatcher by selector value, so renaming
necessarily reorders dispatcher blocks and shifts every jump destination. A literal
opcode-sequence comparison would report a difference for every contract and tell us nothing. The
criterion is equality of the **multiset of basic blocks** with selector and jump-target
immediates masked, plus identical opcode histograms and identical immediates at every other push
width. All 58 contracts pass in both arms. Building this check caught three real renamer bugs
(broken interface overrides, assembly references, struct named arguments), which is the argument
for keeping it as a gate rather than a formality.

Arm A reproduces the shipped benchmark: solc 0.4.26 with no optimizer targeting byzantium
reproduces all 58 shipped binaries byte-for-byte apart from the trailing metadata hash, which
changes because comments are stripped and is never executed. Arm A therefore doubles as a sanity
check on the whole pipeline.

## Detection criterion, and why it had to be remapped

The published numbers count a contract as detected only when the fuzzer raises an integer-bug
alarm **at the address of the catalogued vulnerable instruction** (`plot_b1_cve.py` reading
`B1-cve.csv`), not merely somewhere in the contract. Using the looser criterion inflates the
counts, because the fuzzer does sometimes find an overflow elsewhere; using this one reproduces
all 14 rows of the paper's RQ2 table exactly.

Those addresses were recorded on the original bytecode, and renaming shifts every one of them.
Applied naively across arms, the criterion reports a collapse from ~49 detections to ~2 in arms
B and C — pure addressing artefact, since the alarms are still there. `mapcve.js` carries each
catalogued address across to its counterpart using the same canonicalization that proves the
arms equivalent: it locates the basic block containing the vulnerable instruction and takes the
address of the instruction at the same index in the matching block of the target arm. All 81
addresses map in both arms, with no ambiguity, and in all 116 contract-arm pairs the mapped
addresses land on the same `ADD`, `MUL` or `SUB` instruction.

## Campaigns

2,610 trials: 3 arms x 3 configurations (SmartChat with GPT-4.1-mini seeds, SmartChat with
Llama3.3-70B seeds, Smartian with its own data-flow seeds) x 58 contracts x 5 repetitions, each
a 1-hour campaign in its own container pinned to one dedicated physical core, as in the
published setup.

**Five independent seed sets per arm, one per repetition.** The paper reuses a single seed set
across its five repetitions, so those repetitions capture only the fuzzer's stochasticity.
Pairing repetition *i* with seed set *i* makes them capture generation stochasticity too, at no
campaign cost. Without it, a difference between arms could be sampling noise in one model
response rather than an effect of the renaming. This deviates from the published protocol, so
the arms are internally comparable while the comparison to the published numbers is looser than
a like-for-like rerun would be. Arm A is regenerated rather than reused, because the model served
behind a given API name changes over time and seeds generated a year apart would confound model
drift with the renaming effect.

Generation follows RQ2: the ABI-only prompt, temperature 0.4, ten test cases of at least four
transactions. One response of 1,740 is unusable — GPT-4.1-mini degenerated into a repeated token
until the output limit in `C_gpt4.1mini_rep2/2018-14063` — and is **not** regenerated: the paper
counts invalid responses against the model, and regenerating only failures, here in the opaque
arm, would erase the kind of effect this experiment measures.

## Excluded contracts

Five of the 58 are dropped **from every arm**, because at least one of their trials did not
complete and a truncated campaign under-reports:

| Contract | Why |
|---|---|
| 2018-10376 | the fuzzer is killed at ~18 min under the 6 GB memory cap (the artifact's own setting), in 32 trials spread over arms A, B and C |
| 2018-13202, 2018-13625 | unhandled exception in Smartian's seed handling, arm C |
| 2018-13220 | same, arm C |
| 2018-14063 | same, arm C; this is the contract whose seed set is empty, and an empty seed directory crashes the fuzzer outright |

Dropping them from all arms alike keeps the comparison symmetric. Keeping them would have
manufactured an A > B, C difference caused by tool limits rather than by memorization, which is
precisely the artefact a reader should be suspicious of. Note that the crashes concentrate in
arm C: seed-induced crashes being more frequent when identifiers are opaque is itself a result,
and it is reported rather than hidden by the exclusion. `e1-trials.csv` gives the state of every
one of the 2,610 trials.

## Results

53 contracts, five repetitions per arm, detection by the criterion above.

| Configuration | A | B | C |
|---|---|---|---|
| SmartChat, GPT-4.1-mini seeds | 48.8 | 47.6 | 47.4 |
| SmartChat, Llama3.3-70B seeds | 48.8 | 47.8 | 47.4 |
| Smartian, data-flow seeds (control) | 48.0 | 48.2 | 47.6 |

Equivalence to arm A within ±5% (TOST, α = 0.05), on detection:

| Configuration | B vs A | C vs A |
|---|---|---|
| GPT-4.1-mini | equivalent (p = 0.028) | not shown (p = 0.070) |
| Llama3.3-70B | equivalent (p = 0.046) | equivalent (p = 0.027) |
| Smartian (control) | equivalent (p = 0.003) | equivalent (p = 0.004) |

Instruction coverage is equivalent within ±5% in every arm and configuration (p = 0.004).

**The control is flat**, which is what makes the rest interpretable: Smartian reads bytecode and
not identifiers, so its results must not move across arms, and they do not.

Two readings need the three statistics together rather than any one of them:

- With Llama3.3-70B, arm C is **significantly worse** than arm A in detection (p = 0.028,
  $\hat{A}_{12} = 0.92$) **and** equivalent to it within ±5% (p = 0.027). Both are true: the
  difference is consistent across repetitions and small in magnitude — 1.4 contracts out of 53.
- Effect sizes on detection reach $\hat{A}_{12} = 0.72$--$0.92$ for A against B and C, which
  sounds large but measures ordering consistency, not magnitude: arm A wins most pairwise
  comparisons of repetitions, by about one contract. Reporting the effect size without the
  absolute difference would overstate it as badly as reporting the p-value alone understates it.

With five repetitions per arm the smallest attainable two-sided p is 0.0079 and
$\hat{A}_{12}$ moves in steps of 0.04; both bound what any test on these data can show.

## What this does and does not establish

It establishes that performance does not depend on the specific identifiers of these contracts:
had the results come from recognizing them, arm B would have degraded, and it does not, with
either model. It also shows that the models do not even need the *meaning* of identifiers, since
arm C costs about one contract — which suggests the ABI's structure carries most of what the
task requires, and leaves little room for memorization to be doing work.

It does not exclude **structural** memorization: a model may have learned the idioms of this
contract population, and the line between that and legitimate domain knowledge is genuinely
blurred. It is also confined to Bench58 and to detection; the generalization question is
addressed separately by the post-cutoff sample (see `E3-protocol.md`).

## Infrastructure notes

Campaigns ran on a two-socket Xeon Gold 6338 node, 32 trials in parallel pinned to the physical
cores of one socket with memory bound to the same NUMA node, leaving the second socket and the
SMT siblings idle: time to detection is one of the metrics, and using all 64 cores would have
made trials on different sockets see different memory latencies.

Three environment-specific findings, recorded because they cost time and would cost it again:

- Reading the bytecode from a bind mount backed by an NFS home makes the fuzzer report **zero
  executions with no error at all**. Trials therefore copy their inputs into the container,
  work in a container-local directory, and copy the results out at the end.
- The replayer re-reads every test case, so with the output directory on NFS it took minutes
  instead of seconds.
- Batch 2 stopped 519 trials short when the process's working directory vanished under it,
  presumably an NFS remount: `os.getcwd()` began failing, and because that call sat outside the
  worker's exception handler all 32 workers died at once. The runner now resolves every path at
  startup, tolerates a failing trial without losing the worker, and runs from local disk. The
  campaign was resumed without loss, since finished trials are skipped, but the interval between
  repetitions is consequently not uniform.
