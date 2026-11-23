# Briefing — g0

Feasibility unit counts for HaDukeN //\\\\// HGNLLM Experiments. This iteration records metadata. It does not hold the text that was counted, a tokenizer table, or weights.

## OODA

### Observe

On the valid split the feasibility build used 181,544 units against 249,887 for GPT-2 BPE: 27.35% fewer. Byte fallback was 0.048%. The build took 0.093 CPU-hours and peaked at 4.83 GB. The vocabulary size was 65,536. The record is `results/g0.json`.

### Orient

The written G0 cap is V ≤ 64,000. 65,536 is 2^16. The harness does not treat that as inside the cap. Four G0 limits hold. The vocabulary line does not. This iteration does not contain an hnt64k / struct32k / sw64k pilot, and it does not claim a G1 result.

### Decide

G0 is partial. Stop the vocabulary claim at the recorded size. Do not describe 65,536 as meeting a 64,000 cap.

### Act

A later iteration that wants G0 to pass has to build at V ≤ 64,000 and register that record. A later iteration that wants G1 has to bring the three named arms, three seeds, and raw test bytes. That work is not on this branch.

## 1. Situation

The feasibility build already produced unit counts, a fallback rate, a build time, a peak memory figure, and a vocabulary size. Those five numbers are the friendly picture for this branch. The corpus they were counted on stays on the machine that counted it. Nothing in `results/` is that corpus.

## 2. Mission

Record the feasibility G0 metadata so a reviewer can see that the unit, fallback, CPU-hour, and RAM limits hold, and that V=65,536 fails the written cap of 64,000.

## 3. Execution

The method is the comparison already stored in `results/g0.json`: valid-split unit counts against GPT-2 BPE, plus the build's CPU-hours, peak RAM, byte-fallback percent, and vocabulary size. `MODE_METHOD.md` names the mode and the single gate this iteration may speak to. No training loop ran for this record, and no bits-per-byte figure is attached.

## 4. Sustainment

The recorded build used 0.093 CPU-hours and 4.83 GB peak. This branch does not store the build's output weights, the merge table, or the machine path the build ran on. Recomputing the counts requires the corpus, which is not here.

## 5. Command & Signal

Branch `iter/g0`, cut from `main` at `b9a71763a3caba217ad7707bd12e42b94d05a882`. The public index on `main` is `iterations/INDEX.md`. The overview briefing links this branch. There is no ledger head on this branch: the record is metadata, not a sealed study store. A sealed study uses `sheaf-verify anchor` on the machine that holds the key, and that head is copied by hand.
