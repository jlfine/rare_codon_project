'''
HMM codon randomizer

purpose: uses custom HMM to simulate codon randomized ORFs from specified conditional distributions of codons given hidden genomic states 
(regions of GC-richness, entropy-richness, etc.). 

inputs:
--seq: path to a .csv file containing human ORF sequences (must include a column 'assembled_ORF' with each ORF sequence)
--seed: random seed for reproducibility
--out: output directory, which should contain:
    - transition_matrices_HMM_naive_clusters.json
    - emission_matrices_HMM_naive_clusters.json
    - stationary_dist_HMM_naive_clusters.json

outputs:
1. reverse-translated ORFs for each cluster and the naive model
2. .csv summary file (monte_carlo_summary_HMM.csv) logging number of unique k-mers and RCPGs for each cluster

example usage:

python "/Users/jacobfine/Library/CloudStorage/OneDrive-Personal/U of T 2022-2023/Blencowe/Jan_2024_Blencowe/the_paper/final_code/final_copies/final_final/HMM_parameters/hmm_simulation.py" \
  --seq "/Users/jacobfine/Library/CloudStorage/OneDrive-Personal/U of T 2022-2023/Blencowe/Jan_2024_Blencowe/NOV_2024_reanalysis/sequence_analysis/gencode_v43_longest_orfs_assembled_nov_29-Jacob’s MacBook Pro.csv" \
  --seed 42 \
  --out "/Users/jacobfine/Library/CloudStorage/OneDrive-Personal/U of T 2022-2023/Blencowe/Jan_2024_Blencowe/the_paper/final_code/final_copies/final_final/HMM_parameters/hmm_results_test/"

"""
'''

import pandas as pd
import numpy as np
import argparse
import csv
import os
import random
import math
import string
from collections import Counter



# define the arguments for input files
parser = argparse.ArgumentParser(description="")

parser.add_argument("--seed", type=int, required=True, help="Random seed value")
parser.add_argument("--out", type=str, required=True, help="Path to output CSV file")
parser.add_argument("--seq", type=str, required=True, help="Path to input sequence CSV file")

args = parser.parse_args()

# sets random seed
s = args.seed
random.seed(s)

# confirms seed
print('Configuring HMM... with random seed =', s)


seq_col = 'Sequence'
seq_file = pd.read_csv(args.seq)
seq_file['gene'] = seq_file['name']

# seq_file = seq_file.iloc[:200,:] # for testing

directory = args.out
seq_file[seq_col] = seq_file['assembled_ORF']  # gets the previous seq file that underwent orf analysis


# # read jsons for transition, emissions and stationary disttributions
def json_compatible_to_df(obj):
    if isinstance(obj, dict):
        if obj.get('__is_dataframe__') and 'data' in obj:
            return pd.DataFrame(**obj['data'])
        else:
            return {k: json_compatible_to_df(v) for k, v in obj.items()}
    else:
        return obj

import json


# loads distributions
with open(directory + 'transition_matrices_HMM_naive_clusters.json', 'r') as f:
    global_tms = json_compatible_to_df(json.load(f))

with open(directory + 'emission_matrices_HMM_naive_clusters.json', 'r') as f:
    amino_codon_freq_by_cluster = json_compatible_to_df(json.load(f))

with open(directory + 'stationary_dist_HMM_naive_clusters.json', 'r') as f:
    stationary_dists = json_compatible_to_df(json.load(f))



# the genetic code dict
code = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W'}


# human codon frequencies obtained from hg38 http://gtrnadb.ucsc.edu/genomes/eukaryota/Hsapi38/Hsapi38-summary-codon.html
human_frqs_hg38 = {'GCT': 1.84,
'GCC': 2.77,
'GCG': 0.74,
'GCA': 1.58,
'GGT': 1.08,
'GGC': 2.22,
'GGG': 1.65,
'GGA': 1.65,
'CCT': 1.75,
'CCC': 1.98,
'CCG': 0.69,
'CCA': 1.69,
'ACT': 1.31,
'ACC': 1.89,
'ACG': 0.61,
'ACA': 1.51,
'GTT': 1.10,
'GTC': 1.45,
'GTG': 2.81,
'GTA': 0.71,
'TCT': 1.52,
'TCC': 1.77,
'TCG': 0.44,
'TCA': 1.22,
'AGT': 1.21,
'AGC': 1.95,
'CGT': 0.45,
'CGC': 1.04,
'CGG': 1.14,
'CGA': 0.62,
'AGG': 1.20,
'AGA': 1.22,
'CTT': 1.32,
'CTC': 1.96,
'CTG': 3.96,
'CTA': 0.72,
'TTG': 1.29,
'TTA': 0.77,
'TTT': 1.76,
'TTC': 2.03,
'AAT': 1.70,
'AAC': 1.91,
'AAG': 3.19,
'AAA': 2.44,
'GAT': 2.18,
'GAC': 2.51,
'GAG': 3.96,
'GAA': 2.90,
'CAT': 1.09,
'CAC': 1.51,
'CAG': 3.42,
'CAA': 1.23,
'ATT': 1.60,
'ATC': 2.08,
'ATA': 0.75,
'ATG': 2.20,
'TAT': 1.22,
'TAC': 1.53,
'TGA': 0.16,
'TAG': 0.08,
'TAA': 0.10,
'TGT': 1.06,
'TGC': 1.26,
'TGG': 1.32}


# normalizes the freqs
human_frqs_hg38 = {key: round(frq/100,4) for key, frq in human_frqs_hg38.items()}

# probability distribution of codons conditioned on amino acids P(C|A)

amino_codon_freq_dict_INITIAL = {'I': {'ATA': 0.1693, 'ATC': 0.4695, 'ATT': 0.3612}, 
                        'M': {'ATG': 1.0}, 
                        'T': {'ACA': 0.2838, 'ACC': 0.3553, 'ACG': 0.1147, 'ACT': 0.2462},
                        'N': {'AAC': 0.5291, 'AAT': 0.4709},
                        'K': {'AAA': 0.4334, 'AAG': 0.5666}, 
                        'S': {'AGC': 0.2404, 'AGT': 0.1492, 'TCA': 0.1504, 'TCC': 0.2182, 'TCG': 0.0543, 'TCT': 0.1874},
                        'R': {'AGA': 0.2152, 'AGG': 0.2116, 'CGA': 0.1093, 'CGC': 0.1834, 'CGG': 0.2011, 'CGT': 0.0794},
                        'L': {'CTA': 0.0719, 'CTC': 0.1956, 'CTG': 0.3952, 'CTT': 0.1317, 'TTA': 0.0768, 'TTG': 0.1287},
                        'P': {'CCA': 0.2766, 'CCC': 0.3241, 'CCG': 0.1129, 'CCT': 0.2864},
                        'H': {'CAC': 0.5808, 'CAT': 0.4192}, 
                        'Q': {'CAA': 0.2645, 'CAG': 0.7355},
                        'V': {'GTA': 0.117, 'GTC': 0.2389, 'GTG': 0.4629, 'GTT': 0.1812}, 
                        'A': {'GCA': 0.228, 'GCC': 0.3997, 'GCG': 0.1068, 'GCT': 0.2655}, 
                        'D': {'GAC': 0.5352, 'GAT': 0.4648}, 
                        'E': {'GAA': 0.4227, 'GAG': 0.5773}, 
                        'G': {'GGA': 0.25, 'GGC': 0.3364, 'GGG': 0.25, 'GGT': 0.1636}, 
                        'F': {'TTC': 0.5356, 'TTT': 0.4644}, 
                        'Y': {'TAC': 0.5564, 'TAT': 0.4436}, 
                        '*': {'TAA': 0.2941, 'TAG': 0.2353, 'TGA': 0.4706}, 
                        'C': {'TGC': 0.5431, 'TGT': 0.4569}, 
                        'W': {'TGG': 1.0}}


# k-l divergence calculator to define information gain from codon usage
def codon_KL(seq) -> int:
    # breaks the sequence up into codons
    codons = [seq[i:i+3] for i in range(0,len(seq),3)]
    # gets the polypepitde for that codon
    pp = [code[codon] for codon in codons]

    # gets the counts of codons and amino acids in the sequence
    codon_counts = dict(Counter(codons))
    amino_counts = dict(Counter(pp))

    # makes counts into a dict
    codon_counts = {k: v for k, v in codon_counts.items()}
    amino_counts = {k: v for k, v in amino_counts.items()}

    # gets the total number of amino acids
    total_amino = sum(amino_counts.values())
    KL_divergence = 0  # sets initial k-l div to zero
    for amino in amino_counts.keys():
        # normalizes probs of aminos
        prob_amino = amino_counts[amino]/total_amino

        # gets the expected value of that amino prob
        prob_codon_given_this_amino_pmf_exp = amino_codon_freq_dict[amino]

        # gets all the codons for that amino acid
        codons_for_this_amino = list(amino_codon_freq_dict[amino].keys())

        # gets the observed condition probabileis of the codons P(c|a)
        prob_codon_given_this_amino_pmf_obs = {codon: count/amino_counts[amino] for codon, count in codon_counts.items()
                                               if codon in codons_for_this_amino}
        # the amino acid level k-l; need to sum over all aminos and their probs for the total k-l
        KL_amino = 0
        # for each codon in the seq
        for codon in codons_for_this_amino:
            # gets the obs and expected values of each codon
            P_codon_given_amino_OBS = prob_codon_given_this_amino_pmf_obs[codon] if codon in prob_codon_given_this_amino_pmf_obs.keys() else 0 # accounts for if some codons not observed
            P_codon_given_amino_EXP = prob_codon_given_this_amino_pmf_exp[codon]
            # applies the inner sum of the k-l divergence, getting plog(p/q)
            KL_amino += P_codon_given_amino_OBS*(math.log2(P_codon_given_amino_OBS/P_codon_given_amino_EXP)) if P_codon_given_amino_OBS !=0 else 0  # accounts for 0log0
            
        # sums over all the amino acids 
        KL_divergence += prob_amino*KL_amino

    return KL_divergence


# counts the number of rare codons in a sequence; given the list of rare codons
def rare_codon_count_0(sequence) -> int:  
    rare_codons = ['TCG', 'CGT', 'ACG', 'CGA', 'CCG', 'GTA', 'CTA', 'GCG', 'ATA','TTA']  # codons with less than 1% frequency in the human ORFeome.

    seq = [sequence[i:i + 3] for i in range(0, len(sequence), 3)]
    # sums the number of rare codons in the seq
    rare_count = sum(codon in rare_codons for codon in seq)
    
    return rare_count
    
def GC(sequence):
    if len(sequence) > 0:
        # counts number of C and G in the sequence
        gc_count = sequence.count('G') + sequence.count('C')
        # gets the proportion of GC in the sequence
        gc_fraction = round(gc_count / len(sequence), 3)
        return gc_fraction
    else:
        return np.nan


def CpG(sequence):
    # ensures sequence is valid
    if len(sequence) > 0:
        # count number of CG dinucloetides
        cpg_count = sequence.count('CG')
        # get prob of CG
        cpg_fraction = round(cpg_count / len(sequence), 3) if len(sequence) > 1 else 0
        return cpg_fraction
    else:
        return np.nan


def shannon_entropy_nuc(sequence):
    length_seq = len(sequence) # gets seq length
    shannon_entropy = 0  # initializes entropy to zero 
    bases = ['A','T','C','G'] # alphabet of bases
    for base in bases:
        p = sequence.count(base)/length_seq  # prob of each base

        if p != 0:  # makes sure p isn't zero to avoid 0log0
            logp = math.log2(p)
            plogp = p*logp  

        else: 
            plogp = 0  # set plogp to zero
        shannon_entropy = shannon_entropy + (plogp) # updates entropy this way
        
    shannon_entropy = -round(shannon_entropy,3) # inverts sign according to SE formula
    return shannon_entropy


# class to process codon-related information
class CodonFunctions:
    def __init__(self, seq_file, code, codon_freqs):
        """
        params:
        - seq_file: df with at least a 'Sequence' and 'gene' column.
        - code: the genetic code, a mapping from codon onto amino acid
        - codon_freqs: a dicitonary mapping each codon onto its frequency in human ORFs
        """
        self.seq_file = seq_file
        self.code = code
        self.codon_freqs = codon_freqs
        self.amino_codon_freq_dict = self._build_amino_codon_freq_dict()


    def _build_amino_codon_freq_dict(self):
        """
        returns the dict of P(c|a)
        """
        amino_codon_freq = {}
        for codon, aa in self.code.items():
            if aa not in amino_codon_freq:
                amino_codon_freq[aa] = {}
            amino_codon_freq[aa][codon] = self.codon_freqs[codon]

        for aa, codons in amino_codon_freq.items():
            total = sum(codons.values())
            for codon in codons:
                amino_codon_freq[aa][codon] = round(codons[codon] / total, 4)

        return amino_codon_freq

    def translate_orf(self, orf):
        """
        translates a given codon sequence to its polypeptide
        """
        peptide = ''
        for i in range(0, len(orf) - 2, 3):
            codon = orf[i:i+3]
            # accounts for cases of imperfect sequences
            if 'N' in codon or len(codon) != 3:
                peptide += 'X'
            else:
                peptide += self.code.get(codon, 'X')
        return peptide

    def reverse_translate_peptide(self, peptide):
        """
        Applies probablistic reverse translation (PRT) the codon based on P(codon | amino acid)
        """
        seq = ''
        for aa in peptide:
            if aa != 'X' and aa in self.amino_codon_freq_dict:
                codons = self.amino_codon_freq_dict[aa]
                codon = random.choices(
                    list(codons.keys()),
                    weights=list(codons.values()),
                    k=1
                )[0]
                seq += codon
        return seq

    def add_polypeptide_column(self):
        """
        constructs the ppt column in the df, based on translating each sequence
        """
        self.seq_file['Polypeptide'] = self.seq_file['Sequence'].apply(self.translate_orf)

    def add_random_orf_column(self):
        """
        applys the PRT to generate a random sequence column for each polypeptide
        """
        if 'Polypeptide' not in self.seq_file:
            self.add_polypeptide_column()
        self.seq_file['Sequence_Random'] = self.seq_file['Polypeptide'].apply(self.reverse_translate_peptide)

    # gets only the gene-sequence pairs (since thats what we ultimately care about here)
    def get_seq_file_slice(self, columns=['gene', 'Sequence_Random']):
        return self.seq_file[columns].copy()


# constructs class
codon_funcs = CodonFunctions(seq_file=seq_file, code=code, codon_freqs=human_frqs_hg38)
codon_funcs.add_polypeptide_column()
codon_funcs.add_random_orf_column()

# creates ORF-wide P(c|a). Different ORF regions defined by the HMM will have different P(c|a,z) where z is each hidden state.
amino_codon_freq_dict = codon_funcs.amino_codon_freq_dict
seq_file_slice = codon_funcs.get_seq_file_slice()



class ORF_reverse_translator_HMM_by_gene:
    def __init__(self, seq_file, amino_codon_freq_dict_z_DICT, global_transition_matrix, stationary_dists, cluster_col):
        """
        runs the actual HMM-based PRT.

        inputs:
        - seq_file: df containing a 'Polypeptide' column and a 'gene' column
        - amino_codon_freq_dict_z_DICT: contains the values of P(c|a,z), where z is each state.
        - global_transition_matrix: dict of P(new state | previous state), a.k.a P(z_{i+1}|z_i)
        - stationary_dists: used for sampling states based on their ORF-wide prior densities, P(z)
        - cluster_col: the name of the HMM setting that we're using
        """
        self.seq_file = seq_file
        self.codon_dict_z = amino_codon_freq_dict_z_DICT
        self.transition_matrix = global_transition_matrix

        # extracts the initial state distribution
        initial_state_dist = stationary_dists[cluster_col]
        self.initial_states = list(initial_state_dist.keys())
        self.initial_weights = list(initial_state_dist.values())

        # for keeping track of the state transition path. Assigns letter to each state for that.
        self.z_to_letter = {
            z: string.ascii_uppercase[i % 26]
            for i, z in enumerate(self.initial_states)
        }
        self.cluster_col = cluster_col

    def sample_initial_state(self):
        # samples an initial state from the prior state distribtion we gave it
        initial_state = random.choices(self.initial_states, weights=self.initial_weights, k=1)[0]
        return initial_state

    def sample_next_state(self, current_z):
        # raise error if the current state is not present
        if current_z not in self.transition_matrix:
            raise ValueError(f"state '{current_z}' is not found in transition matrix provided")

        # gets the full P(z_{i+1}|z_i) for the current state, and converts it to a dict
        probs = self.transition_matrix[current_z]
        if isinstance(probs, pd.Series):
            probs = probs.to_dict()

        # samples the next state
        next_state = random.choices(
            list(probs.keys()), weights=list(probs.values()), k=1
        )[0]
        return next_state

    # randomly reverse translates polypeptides using the HMM for the entire seq_file
    def reverse_translate_polypeptides(self):
        polypeptides = self.seq_file['Polypeptide']
        # keeps track of the HMM-randomized ORFs
        translated_orfs = []
        # keeps track of the state paths used
        state_paths = []

        # iterates through each polypeptide
        total_polypeptides = len(polypeptides) # number of polypeptides to translate
        for i, poly in enumerate(polypeptides):

            print(f"[{i+1}/{total_polypeptides}] translating ORF for", end='\r')  # keeps track of progress

            orf = ''
            # the sequence of states for that seq. recall that a given codon is associated with a particular state
            state_seq = []

            # samples the initial state
            current_z = self.sample_initial_state()
            # appends the letter associated with that state to the list for that state
            state_seq.append(self.z_to_letter[current_z])

            # for each amino acid in the polypeptide
            for aa in poly:
                # makes sure its a legal amino acid, and the current state is legal
                if (
                    aa != 'X'
                    and current_z in self.codon_dict_z
                    and aa in self.codon_dict_z[current_z]
                ):
                    # gets the P(c|a,z); samples one new codon from that dist!
                    codon_weights = self.codon_dict_z[current_z][aa]
                    codon = random.choices(
                        list(codon_weights.keys()),
                        weights=list(codon_weights.values()),
                        k=1
                    )[0]
                    orf += codon
                else:
                    # otherwise appends the placeholder codon
                    orf += 'NNN'

                # samples the next state
                current_z = self.sample_next_state(current_z)
                # keeps track of the previous state now as it continues. does it for each codon in the polypeptide
                state_seq.append(self.z_to_letter[current_z])

            # adds the new translated polypeptide to the list of ORFs
            translated_orfs.append(orf)
            # keeps track of the state path
            state_paths.append(''.join(state_seq))
        print() # adds new line

        # updates the seq_file accoridglu
        self.seq_file['Sequence_Random'] = translated_orfs
        self.seq_file['State_Path'] = state_paths

print('Starting HMM... with random seed =', s)

reverse_translated_results_by_cluster = {}

# goes through each HMM setting and implements the functinos
for i, cluster_col in enumerate(stationary_dists.keys()):
    print(f"Processing group {i+1} of {len(stationary_dists.keys())}, labelled '{cluster_col}'")
    # gets P(c|a,z) for the current HMM setting
    amino_codon_freq_dict_z_DICT_i = amino_codon_freq_by_cluster[cluster_col]
    # gets transition matrix, P(z_{i+1}|z_i) 
    global_transition_matrix_i = global_tms[cluster_col]
    # gets the initial state disitrbuiotn P(z)
    initial_state_dist_i = stationary_dists[cluster_col]

    # constructs the reverse translator on the current HMM setting
    reverse_translator = ORF_reverse_translator_HMM_by_gene(
        seq_file=seq_file.copy(),
        amino_codon_freq_dict_z_DICT=amino_codon_freq_dict_z_DICT_i,
        global_transition_matrix=global_transition_matrix_i,
        stationary_dists=stationary_dists,
        cluster_col=cluster_col
    )
    reverse_translator.reverse_translate_polypeptides()
    reverse_translated_results_by_cluster[cluster_col] = reverse_translator.seq_file.copy()


# adds the result for the no-HMM case
reverse_translated_results_by_cluster['native'] = seq_file_slice


# from the HMM-generated ORFs, we now want to find how many rare codon patches there were.
# first, this implements the sliding window analysis
def generate_kmers_from_orf(df, k=45, step=3):
    rows = []
    
    for _, row in df.iterrows():

        gene = row['gene']

        orf = row['full_ORF']
        orf_len = len(orf)
        
        for pos in range(0, orf_len - k + 1, step):
            kmer = orf[pos:pos + k]
            if len(kmer) == k:
                rows.append({
                    'gene': gene,
                    'full_ORF': orf,
                    'position': pos,
                    'kmer': kmer,
                    'percentile': pos / orf_len  # this matches your meta-kmer pipeline
                })
    
    return pd.DataFrame(rows)

# this is now the script to merge adjancet k-mers
# given a list of start coordinates (where each k-mer starts in the ORF), aim is to join together numbers that are close into one interval, 
# so we can use that interval to slice the ORF sequence
def tuple_finder(positions) -> list: # we take a list of positions; i.e., the position that each k-mer starts in the ORF sequence.
    if not positions:
        return []

    result = []  # the result will be a list of tuples
    current_list = [positions[0]]  # we have a working list of a given set of start coords that fall in a 45 nt interval

    for pos in positions[1:]:  # then go through each other position
        if pos - current_list[-1] <= 45: # so long as they are not more than or equal to 45 nt away from the last position
            current_list.append(pos)  # join that position to the current list
        else:  # otherwise, get the first and last position (plus 45, so its the end index)
            result.append((current_list[0], current_list[-1]+45))  
            current_list = [pos]  # now, reset the current list to the current one (since it was outside 45 nt)
    
    # appends the last current_list
    if current_list: 
        result.append((current_list[0], current_list[-1]+45))

    return result  # returns a lit of tuples




# uses the tuple finder to process the k-mers (to merge them)

def process_kmers(kmer_data, tuple_finder):
    # groups the data by the gene name col
    grouped_data = kmer_data.groupby('gene')
    # makes  alist of the rows
    row_list = []

    for gene_name, group in grouped_data:
        positions_list = group['position'].tolist()
        # for the list of positions associated with each gene; 
        # i.e., the list where all its rare codon patches start, get their (start,end) from the tuple finder
        meta_kmer_positions = tuple_finder(positions_list)
        # get the sequence of the fulll ORF for that gene
        sequence_ORF = group['full_ORF'].iloc[0]

        # iterates through each (start,end) representing a different merged k-mer slice.
        for start, end in meta_kmer_positions:
            meta_kmer = sequence_ORF[start:end] # gets the merged k-mer; the 'meta_kmer'
            count_rare = rare_codon_count_0(meta_kmer) # counts the number of rare codons 
            prop_rare = count_rare / (len(meta_kmer) / 3)  # proporiton rare codons
            length_in_nt = len(meta_kmer) # length in nt of the k-mer

            row_dict = {
                'ensg_name': gene_name,
                'meta_kmer_id': f"{gene_name}_{(start, end)}", # a unique id we can assign based on the genename, start and end of each k-mer
                'meta_kmer': meta_kmer,
                'count_rare_codons': count_rare,
                'proportion_rare_codons': prop_rare,
                'length_in_nt': length_in_nt,
                'percentile': start / len(sequence_ORF), # gets where the k-mer is in the ORF
                'sequence_ORF_original': sequence_ORF,
            }

            row_list.append(row_dict)

    return pd.DataFrame(row_list) # returns a df of the processed rows


# now, we want to keep track of the number of rare codon patches and RCPGs obtained in each HMM setting.
df_kmer_processed_i_dict = {}
cluster_cols = reverse_translated_results_by_cluster.keys() # goes throguh each one
for i, col in enumerate(cluster_cols):
    print(f'background {i+1} of {len(cluster_cols)}:')
    seq_file_result = reverse_translated_results_by_cluster[col].copy()


    seq_file_result['full_ORF']=seq_file_result['Sequence_Random']
    if 'name' in seq_file_result.columns: # to ensure the gene col is correct
        seq_file_result['gene']=seq_file_result['name'].copy() 

    df_kmers_generated = generate_kmers_from_orf(seq_file_result, k=45, step=3)
    df_kmers_generated['rare_codon_count_0'] = df_kmers_generated['kmer'].apply(rare_codon_count_0)

    # 8 or more rare codons per k-mer
    df_kmers_copy_human_rare_8_rare = df_kmers_generated[df_kmers_generated['rare_codon_count_0']>=8]

    df_kmers_copy_human_top_rare_grouped = df_kmers_copy_human_rare_8_rare.groupby('gene')

    # uses function to merge adjancet rare codon patch k-mers
    df_kmers_processed = process_kmers(
        kmer_data=df_kmers_copy_human_rare_8_rare, 
        tuple_finder=tuple_finder)
    print(col)
    print(len(df_kmers_processed))
    print('')
    df_kmer_processed_i_dict[col] = df_kmers_processed

    # the values of the test statistics (number of rare codon patches and RCPGs under the HMM simulations)
    k = len(df_kmers_processed.drop_duplicates('meta_kmer_id'))
    g = len(df_kmers_processed.drop_duplicates('ensg_name'))

    # prints the result for refernece.
    print('seed =', s)
    print('cluster',col)
    print('kmers =', k)
    print('genes =', g)

    # the file path to save the results
    csv_path = directory+  "monte_carlo_summary_HMM.csv"
    write_header = not os.path.exists(csv_path)

    # updates results with values of the test statistics
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['seed', 'cluster','k', 'g'])
        if write_header:
            writer.writeheader()
        writer.writerow({'seed': s, 'cluster': col, 'k': k, 'g': g}) 
    