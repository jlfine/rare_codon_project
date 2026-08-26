#!/usr/bin/env python3
# rare codon patch conservation analysis
# for a pair of species (human-x) get percent of patches observed vs expected
# accross all human RCPGs


import os
import sys
import json
import gzip
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
import time


# write the print statements to a log file
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Zoonomia multiple codon alignments to analyze
input_dir = "/human_hg38_reference/MultipleCodonAlignments"
# to save the outputs of the simulation
default_output_dir = "outputs_codon_analysis/transition_matrix_results"
# df of the rare codon patches and info
default_master_csv = "orfs/df_master.csv"


rare_codons = {'TCG', 'CGT', 'ACG', 'CGA', 'CCG', 'GTA', 'CTA', 'GCG', 'ATA', 'TTA'}
## hypermarameters for analuysis
window_size = 15
step_size = 1
min_rare = 8
max_gap_fraction = 0.2 ## filter for genes with too many gaps
n_sims_default = 500

# 65 states; 64 codons, plus gap
codons_64 = sorted([f"{a}{b}{c}" for a in "ACGT" for b in "ACGT" for c in "ACGT"])
codons = codons_64 + ['GAP']
codon_to_idx = {c: i for i, c in enumerate(codons)}


# split a sequence into codons
def extract_codons(seq):
    return [seq[i:i+3].upper() for i in range(0, len(seq) - 2, 3)]

# check a codon is 3 valid nucleotides
def is_valid_codon(codon):
    return len(codon) == 3 and all(c in 'ACGT' for c in codon)

# list the gzipped alignment files in a directory
def _list_alignment_files(in_dir=input_dir):
    return sorted([
        f for f in os.listdir(in_dir)
        if f.endswith('.fasta.gz') or f.endswith('.fa.gz')
    ])

# take the gene symbol out of an alignment filename
def _gene_symbol_from_filename(fname):
    base = fname.replace('.fasta.gz', '').replace('.fa.gz', '')
    parts = base.split('.')
    return parts[1] if len(parts) >= 2 else base

# drops columns where both codons are gaps; removing noice; to compare ref to target for each gene
def _extract_pairwise(ref_codons_aln, tgt_codons_aln):
    pairwise = [(r, t) for r, t in zip(ref_codons_aln, tgt_codons_aln)
                if not (not is_valid_codon(r) and not is_valid_codon(t))]
    ref_codons = [r if is_valid_codon(r) else 'GAP' for r, t in pairwise]
    tgt_codons = [t if is_valid_codon(t) else 'GAP' for r, t in pairwise]
    return ref_codons, tgt_codons


# asks if a given window is a rare codon patch window; if it has 8 or more rare codons
def count_observed_patches(seq_codons, rare_set=rare_codons,
                           win=window_size, step=step_size,
                           thresh=min_rare):
    n = len(seq_codons)
    if n < win:
        return 0
    is_rare = np.array([c in rare_set for c in seq_codons])
    count = 0
    for k in range(0, n - win + 1, step):
        if is_rare[k:k + win].sum() >= thresh:
            count += 1
    return count


# finds RCPGs in the alignment; makes sure they overlap with our ref set
# key is we only want to analyze the alignments that correspond to RCPGs
def find_human_rcp_genes(rcpgs, in_dir=input_dir, min_patches=1):
    all_files = _list_alignment_files(in_dir)
    rcpgs_set = set(rcpgs)
    ## makes the gene names to their file names
    matched = {}
    for fname in all_files:
        gene = _gene_symbol_from_filename(fname)
        if gene in rcpgs_set:
            matched[gene] = fname

    n_input = len(rcpgs_set)
    n_matched = len(matched)
    n_missing = n_input - n_matched
    missing_genes = rcpgs_set - set(matched.keys())
    ## records how many actually had matches
    print(f"input rcp genes: {n_input}")
    print(f"matched to alignment files: {n_matched}")
    print(f"no alignment file found: {n_missing} ({n_missing/n_input*100:.1f}%)")
    if missing_genes and n_missing <= 20:
        print(f"  missing: {sorted(missing_genes)}")

    rcp_files = []
    non_rcp_files = []
    no_ref = 0
    too_short = 0
    t0 = time.time()
    ## iterates through each gene-file name pair
    for i, (gene, fname) in enumerate(sorted(matched.items())):
        if (i + 1) % 500 == 0:
            print(f"  scanning {i+1}/{n_matched}... {len(rcp_files)} rcp genes so far")

        fpath = os.path.join(in_dir, fname)
        try:
            ref_seq = None
            with gzip.open(fpath, 'rt') as handle:
                ## reference is the human genome
                for rec in SeqIO.parse(handle, 'fasta'):
                    if rec.id == 'REFERENCE':
                        ref_seq = str(rec.seq).upper()
                        break
        except Exception:
            continue

        if ref_seq is None:
            no_ref += 1
            continue

        ## ensure the gene has more than enough codons
        
        ref_codons = [c for c in extract_codons(ref_seq) if is_valid_codon(c)]
        if len(ref_codons) < window_size:
            too_short += 1
            continue
        ## threshold to call something a rare codon patch gene
        n_patches = count_observed_patches(ref_codons)
        if n_patches >= min_patches:
            rcp_files.append((gene, fname, n_patches))
        else:
            non_rcp_files.append((gene, fname))

    elapsed = time.time() - t0
    print(f"\ndone scanning {n_matched} files in {elapsed:.1f}s")
    print(f"  no reference sequence: {no_ref}")
    print(f"  too short (< {window_size} codons): {too_short}")
    print(f"  confirmed rcp (>= {min_patches} patch): {len(rcp_files)}")
    print(f"  no patches found: {len(non_rcp_files)}")

    return rcp_files, non_rcp_files


# build the 65-state pi_rare vector, p(codon is a rare codon in species s | the specific codon in human)
def build_pi_rare(target_species, gene_file_list, in_dir=input_dir):

    ## for each human state, what fraction are rare in the other speceis?
    rare_counts = np.zeros(65, dtype=np.int64)
    total_counts = np.zeros(65, dtype=np.int64)

    n_found = 0
    n_pairs = 0
    t0 = time.time()

    ## iterate through each gene file
    for i, entry in enumerate(gene_file_list):
        fname = entry[1]
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(gene_file_list)} files, {n_pairs:,} pairs...")

        fpath = os.path.join(in_dir, fname)
        try:
            records = {}
            with gzip.open(fpath, 'rt') as handle:
                for rec in SeqIO.parse(handle, 'fasta'):
                    records[rec.id] = str(rec.seq).upper()
        except Exception:
            continue

        if 'REFERENCE' not in records or target_species not in records:
            continue

        n_found += 1

        ## the codons from the human reference sequence
        ref_codons_aln = extract_codons(records['REFERENCE'])

        ## the codons from the target species (at same respecitve indices)
        tgt_codons_aln = extract_codons(records[target_species])

        ## ensure its the same length
        if len(ref_codons_aln) != len(tgt_codons_aln):
            continue

    
        ref_codons, tgt_codons = _extract_pairwise(ref_codons_aln, tgt_codons_aln)

        ## for each codon pair (ref and target), cadd that codon to the total count
        for rc, tc in zip(ref_codons, tgt_codons):
            idx = codon_to_idx[rc]
            total_counts[idx] += 1

            ## if its a rare count, update the count of that id's rare by 1
            if is_valid_codon(tc) and tc in rare_codons:
                rare_counts[idx] += 1
            n_pairs += 1

    elapsed = time.time() - t0
    print(f"done: {n_found} genes with reference + {target_species}")
    print(f"total positions: {n_pairs:,}")
    print(f"time: {elapsed:.1f}s")

    ## the dictoionaries of the probabilities
    pi_rare = {}
    counts = {}
    for codon in codons:
        idx = codon_to_idx[codon]
        n_r = int(rare_counts[idx])
        n_t = int(total_counts[idx])
        ## normalizes the probs
        pi_rare[codon] = n_r / n_t if n_t > 0 else 0.0
        counts[codon] = [n_r, n_t]

    return pi_rare, counts


# poisson binomial upper tail, p(sum of bernoullis >= threshold)
# implementation of analytical solution to Poisson-Binomial (see Methods for pseudocode and full descirption)
def poisson_binomial_tail(probs, threshold):
    length = len(probs)
    if length == 0 or threshold > length:
        return 0.0
    if threshold <= 0:
        return 1.0
    f = np.zeros(length + 1)
    f[0] = 1.0
    for p in probs:
        for s in range(length, 0, -1):
            f[s] = (1 - p) * f[s] + p * f[s - 1]
        f[0] *= (1 - p)
    return f[threshold:].sum()


# for each gene, compare the observed, analytical, and simulated preservation of human patch windows
def expected_vs_observed_patches(pi_rare, target_species, rcp_gene_list,
                                 in_dir=input_dir,
                                 gap_fraction_max=max_gap_fraction,
                                 run_sim=True, n_sims=n_sims_default,
                                 seed=42):
    rng = np.random.default_rng(seed) if run_sim else None
    results = []
    t0 = time.time()
    n_total = len(rcp_gene_list)
    n_skipped_gaps = 0
    n_skipped_no_patches = 0

    ## loops throught the human rcpgs

    for i, (gene, fname, human_patches_original) in enumerate(rcp_gene_list):
        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        fpath = os.path.join(in_dir, fname)
        try:
            records = {}
            with gzip.open(fpath, 'rt') as handle:
                for rec in SeqIO.parse(handle, 'fasta'):
                    records[rec.id] = str(rec.seq).upper()
        except Exception:
            continue

        if 'REFERENCE' not in records or target_species not in records:
            continue

        ## gets the codons from ref and target species

        ref_codons_aln = extract_codons(records['REFERENCE'])
        tgt_codons_aln = extract_codons(records[target_species])

        if len(ref_codons_aln) != len(tgt_codons_aln):
            continue

        ref_codons, tgt_codons = _extract_pairwise(ref_codons_aln, tgt_codons_aln)
        n = len(ref_codons)

        if n < window_size:
            continue
        ## counts the gaps
        n_ref_gaps = sum(1 for c in ref_codons if c == 'GAP')
        n_tgt_gaps = sum(1 for c in tgt_codons if c == 'GAP')

        # gap filter; for genes above the gap threshold
        gap_frac = max(n_ref_gaps, n_tgt_gaps) / n
        if gap_frac > gap_fraction_max:
            n_skipped_gaps += 1
            continue

        # booleamns for whether eahc codon in human and ref are rare
        ref_is_rare = np.array([c in rare_codons for c in ref_codons])
        tgt_is_rare = np.array([c in rare_codons for c in tgt_codons])
        ## gets the prob at each position for human codons
        pi_vec = np.array([pi_rare.get(c, 0.0) for c in ref_codons])

        # walk windows (only process where human has a patch)
        n_human_patches = 0
        n_target_preserved = 0
        e_preserved = 0.0

        # track which windows are human patches (for simulation)
        human_patch_windows = []

        for k in range(n - window_size + 1):
            if ref_is_rare[k:k + window_size].sum() >= min_rare:
                n_human_patches += 1
                human_patch_windows.append(k)

                # computes positions that are rare codon patch windows
                if tgt_is_rare[k:k + window_size].sum() >= min_rare:
                    n_target_preserved += 1

                # gets the expected number of patch windows from poisson binomial
                e_preserved += poisson_binomial_tail(
                    pi_vec[k:k + window_size], min_rare
                )

        if n_human_patches == 0:
            n_skipped_no_patches += 1
            continue

        # runs a Monte carlo simulation using probabilities from pi_vec to check that analytical result agrees with expected
        sim_mean = None
        sim_std = None
        sim_pvalue = None

        if run_sim and rng is not None:


            draws = rng.random((n_sims, n)) ## a matrix of dim n_sims x n, of uniform random numbers
            ## more efficeint way of doing many simulations for one gene

            ## effective way of labellling codon as rare or not; flips 'biased coin' at each posoition
            sim_rare = draws < pi_vec  

            # for each sim, count how many human-patch windows are also patches in the other species
            sim_preserved = np.zeros(n_sims, dtype=int)
            for k in human_patch_windows:
                ## iterates through each human rare patch window, 
                ## and computes null expectation; the sums of rare codons will determine whether the window is 
                ## a rare patch window or not
                window_sums = sim_rare[:, k:k + window_size].sum(axis=1)
                sim_preserved += (window_sums >= min_rare).astype(int)

            sim_mean = round(sim_preserved.mean(), 4)
            sim_std = round(sim_preserved.std(), 4)
            sim_pvalue = round((sim_preserved >= n_target_preserved).mean(), 6)
        ## stores key results, particularly the pct_preserved_obs and pct_preserved_exp.
        row = {
            'gene': gene,
            'n_pairwise': n,
            'n_ref_gaps': n_ref_gaps,
            'n_tgt_gaps': n_tgt_gaps,
            'gap_fraction_ref': round(n_ref_gaps / n, 4),
            'gap_fraction_tgt': round(n_tgt_gaps / n, 4),
            'human_patch_windows': n_human_patches,
            'target_preserved': n_target_preserved,
            'expected_preserved': round(e_preserved, 6),
            'pct_preserved_obs': round(n_target_preserved / n_human_patches * 100, 2),
            'pct_preserved_exp': round(e_preserved / n_human_patches * 100, 2),
            'ratio_obs_exp': (
                round(n_target_preserved / e_preserved, 4) if e_preserved > 0
                else None
            ),
        }

        if run_sim:
            row['sim_preserved_mean'] = sim_mean
            row['sim_preserved_std'] = sim_std
            row['pct_preserved_sim'] = (
                round(sim_mean / n_human_patches * 100, 2)
                if sim_mean is not None else None
            )
            row['sim_pvalue'] = sim_pvalue
            row['n_sims'] = n_sims

        results.append(row)

    elapsed = time.time() - t0
    print(f"done: {len(results)} rcp genes, {elapsed:.1f}s")
    print(f"  skipped (too gappy): {n_skipped_gaps}")
    print(f"  skipped (no patches in pairwise): {n_skipped_no_patches}")

    return pd.DataFrame(results)


# helps to visualize the pi vector
def summarize_pi_rare(pi_rare, counts):
    total_rare = sum(c[0] for c in counts.values())
    total_all = sum(c[1] for c in counts.values())

    print("pi_rare summary (65 x 1 binary channel vector)")
    print(f"total positions: {total_all:,}")
    print(f"global p(target rare): {total_rare / total_all:.4f}")
    print()
    n_r_gap, n_t_gap = counts['GAP']
    print(f"gap state: pi = {pi_rare['GAP']:.4f}  ({n_r_gap:,} / {n_t_gap:,})")
    print()
    print("all 65 states (sorted by pi, highest first):")
    for c, p in sorted(pi_rare.items(), key=lambda x: -x[1]):
        n_r, n_t = counts[c]
        tag = ""
        if c in rare_codons:
            tag = " <-- rare"
        elif c == 'GAP':
            tag = " <-- gap"
        print(f"  {c:>3s}: pi = {p:.4f}  ({n_r:,} / {n_t:,}){tag}")


# print a summary of the preservation results for a species
def print_results_summary(df, species_name):
    n = len(df)
    total_human = df['human_patch_windows'].sum()
    total_obs = df['target_preserved'].sum()
    total_exp = df['expected_preserved'].sum()
    n_more = (df['target_preserved'] > df['expected_preserved']).sum()

    print(f"\nresults: {species_name}")
    print(f"  rcp genes tested:               {n}")
    print(f"  total human patch windows:       {total_human}")
    print(f"  total preserved in target:       {total_obs}")
    print(f"  total expected preserved:        {total_exp:.1f}")
    if total_exp > 0:
        print(f"  global ratio (obs/exp):          {total_obs / total_exp:.2f}")
    if total_human > 0:
        print(f"  global % preserved (obs):        {total_obs / total_human * 100:.1f}%")
        print(f"  global % preserved (exp):        {total_exp / total_human * 100:.1f}%")
    print(f"  genes with obs > exp:            {n_more} / {n} ({n_more/n*100:.1f}%)")
    print(f"  median % preserved (obs):        {df['pct_preserved_obs'].median():.1f}%")
    print(f"  median % preserved (exp):        {df['pct_preserved_exp'].median():.1f}%")

    if 'sim_preserved_mean' in df.columns:
        total_sim = df['sim_preserved_mean'].sum()
        print(f"\n simulation")
        print(f"  total simulated preserved:       {total_sim:.1f}")
        if total_human > 0:
            print(f"  global % preserved (sim):        {total_sim / total_human * 100:.1f}%")
        print(f"  median % preserved (sim):        {df['pct_preserved_sim'].median():.1f}%")
        print(f"  analytical vs sim correlation:    {df['expected_preserved'].corr(df['sim_preserved_mean']):.4f}")
        print(f"  genes with sim p < 0.05:         {(df['sim_pvalue'] < 0.05).sum()} / {n}")


# to call the pipeline on a particular human-species pair, scalled to all zoonomia
def main():
    parser = argparse.ArgumentParser(
        description="rare codon patch conservation analysis"
    )
    parser.add_argument("species", help="target species, e.g. vs_mm39, vs_panTro6")
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--input_dir", default=input_dir)
    parser.add_argument("--master_csv", default=default_master_csv)
    parser.add_argument("--rcpgs_csv", default=None)
    parser.add_argument("--rcpgs_txt", default=None)
    parser.add_argument("--use_ensg", action="store_true")
    parser.add_argument("--n_sims", type=int, default=n_sims_default,
                        help="simulations per gene (0 or --no_sim to skip)")
    parser.add_argument("--no_sim", action="store_true",
                        help="skip forward simulation")
    parser.add_argument("--max_gap_fraction", type=float, default=max_gap_fraction)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    species = args.species
    run_sim = not args.no_sim and args.n_sims > 0

    # loads gene list
    if args.rcpgs_csv:
        df_genes = pd.read_csv(args.rcpgs_csv)
        rcpgs = list(df_genes['gene'].dropna().unique())
        print(f"loaded {len(rcpgs)} genes from {args.rcpgs_csv}")
    elif args.rcpgs_txt:
        with open(args.rcpgs_txt) as f:
            rcpgs = [line.strip() for line in f if line.strip()]
        print(f"loaded {len(rcpgs)} genes from {args.rcpgs_txt}")
    else:
        if not os.path.exists(args.master_csv):
            print(f"error: master csv not found: {args.master_csv}")
            sys.exit(1)
        df_master = pd.read_csv(args.master_csv)
        if args.use_ensg:
            rcpgs = list(set(df_master['ensg_name_x'].dropna()))
        else:
            rcpgs = list(set(df_master['common_name_x'].dropna()))
        print(f"loaded {len(rcpgs)} genes from {args.master_csv}")

    # output directory
    species_dir = os.path.join(args.output_dir, species)
    os.makedirs(species_dir, exist_ok=True)

    # log of results
    log_path = os.path.join(species_dir, f"log_{species}.txt")
    sys.stdout = TeeLogger(log_path)

    print(f"\nspecies: {species}")
    print(f"simulation: {'on (' + str(args.n_sims) + ' sims)' if run_sim else 'off'}")
    print(f"max gap fraction: {args.max_gap_fraction}")
    print(f"output: {species_dir}\n")

    # step 1, find human RCPG genes
    print("*** step 1: finding human rcp genes ***")
    rcp_files, non_rcp_files = find_human_rcp_genes(rcpgs, in_dir=args.input_dir)
    if len(rcp_files) == 0:
        print("no rcp genes found. exiting.")
        sys.exit(1)

    # step 2, build pi_rare
    print(f"\n*** step 2: building pi_rare (65 states) for {species} ***")
    pi_rare, counts = build_pi_rare(species, rcp_files, in_dir=args.input_dir)

    pi_path = os.path.join(species_dir, f"pi_rare_{species}.json")
    counts_path = os.path.join(species_dir, f"counts_{species}.json")
    with open(pi_path, 'w') as f:
        json.dump(pi_rare, f, indent=2)
    with open(counts_path, 'w') as f:
        json.dump(counts, f, indent=2)
    print(f"  saved: {pi_path}")
    print(f"  saved: {counts_path}")

    summarize_pi_rare(pi_rare, counts)

    # step 3, run simulation
    sim_label = f" + {args.n_sims} sims" if run_sim else ""
    print(f"\n*** step 3: preservation analysis{sim_label} ***")
    df = expected_vs_observed_patches(
        pi_rare, species, rcp_files,
        in_dir=args.input_dir,
        gap_fraction_max=args.max_gap_fraction,
        run_sim=run_sim,
        n_sims=args.n_sims,
        seed=args.seed
    )

    results_path = os.path.join(species_dir, f"results_{species}.csv")
    df.to_csv(results_path, index=False)
    print(f"  saved: {results_path}")

    print_results_summary(df, species)

    print(f"\nall outputs saved to {species_dir}")
    print(f"  {os.path.basename(pi_path)}")
    print(f"  {os.path.basename(counts_path)}")
    print(f"  {os.path.basename(results_path)}")
    print(f"  {os.path.basename(log_path)}")


if __name__ == "__main__":
    main()