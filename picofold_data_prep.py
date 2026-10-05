"""
Build a protein-structure dataset from real proteins: short windows of
sequence + C-alpha coordinates, cut out of experimentally determined structures.

A 16-residue window mostly holds local structure (pieces of
alpha helices, beta strands and loops), which is largely decided by the
window's own sequence, so it's a learnable task for a tiny model.

Recipe:
  1. Download the CATH S40 set: ~31,000 protein domains, no two sharing more
     than 40% sequence identity, so no protein family dominates.
  2. For a random 10% of domains, read the C-alpha atom of every residue.
  3. Split each domain wherever the chain is broken (missing residues).
  4. Cut 16-residue windows, center each at the origin, and save.

Outputs, one window per line:
  train_data.txt and test_data.txt
  each line: domain_id SEQUENCE x1 y1 z1 x2 y2 z2 ... (in Angstroms)

Train and test are split by domain, never by window. Windows from the same
protein overlap, so splitting by window would leak test data into training.

Written by Claude
"""

import os      # os.path.exists
import math    # math.dist
import random  # random.seed, random.random
import tarfile # reading the downloaded .tgz archive
import urllib.request # downloading it
random.seed(42) # Let there be order among chaos

WINDOW = 16            # residues per training example
STRIDE = 8             # start a new window every 8 residues (windows overlap by half)
DOMAIN_FRACTION = 0.1  # use a random 10% of the ~31,000 domains (tens of thousands of windows)
TEST_FRACTION = 0.1    # hold out 10% of the chosen domains for testing

# -----------------------------------------------------------------------------
# Step 1: download the CATH S40 domains (one time)
#
# CATH chops every protein in the Protein Data Bank (PDB) into domains:
# compact units that fold on their own. The "S40" set keeps one representative
# per group of similar sequences. The archive holds one PDB-format text file
# per domain, in a folder called dompdb/. Same file as used by FoldingDiff.
# If the HTTP link fails, the same file is on FTP at:
# ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/

URL = ("http://download.cathdb.info/cath/releases/latest-release/"
       "non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz")
ARCHIVE = "cath-dataset-nonredundant-S40.pdb.tgz"
if not os.path.exists(ARCHIVE):
    print(f"downloading {URL} (large, one time only)...")
    urllib.request.urlretrieve(URL, ARCHIVE)

# -----------------------------------------------------------------------------
# Step 2: read the C-alpha atoms out of a PDB file
#
# PDB is a fixed-column text format, one atom per line, for example:
#
#   ATOM      2  CA  MET A   1      27.340  24.430   2.614  1.00  9.67   C
#
# Columns 13-16 hold the atom name, 17 an "alternate location" flag, 18-20 the
# residue type, 22-27 the chain, residue number and insertion code, and 31-54
# the x, y, z coordinates in Angstroms. Each residue has one C-alpha atom
# ("CA"), the backbone carbon its side chain hangs off, so one C-alpha per
# residue traces the chain's path through space.

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

def read_calphas(lines):
    # Returns a list of (amino_acid_letter, (x, y, z)), in chain order.
    residues, last_residue_id = [], None
    for line in lines:
        if line.startswith('ENDMDL'):
            break # some files hold several models; keep only the first
        if not line.startswith('ATOM') or line[12:16].strip() != 'CA':
            continue # not a C-alpha atom
        if line[16] not in ' A':
            continue # an atom seen in two positions: keep only the first (A)
        residue_id = line[21:27] # chain + residue number + insertion code
        if residue_id == last_residue_id:
            continue # same residue again: skip duplicates
        last_residue_id = residue_id
        letter = THREE_TO_ONE.get(line[17:20], 'X') # X = unusual residue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        residues.append((letter, xyz))
    return residues

# -----------------------------------------------------------------------------
# Step 3: split at chain breaks
#
# Consecutive C-alphas are always 3.8 Angstroms apart (2.9 for a rare "cis"
# bond). A bigger gap means residues are missing from the file, usually a
# floppy loop the experiment couldn't see. Windows must not span a gap, or the
# model would be asked to place residues that were never there.

def split_at_breaks(residues):
    segments, current = [], []
    for letter, xyz in residues:
        if current and math.dist(current[-1][1], xyz) > 4.2:
            segments.append(current) # gap found: close this segment
            current = []
        current.append((letter, xyz))
    if current:
        segments.append(current)
    return segments

# -----------------------------------------------------------------------------
# Step 4: cut windows and write them out

def windows(segment):
    # Yields (sequence, coords) for each WINDOW-long piece of a segment.
    letters = ''.join(letter for letter, _ in segment)
    coords = [xyz for _, xyz in segment]
    for start in range(0, len(segment) - WINDOW + 1, STRIDE):
        seq = letters[start:start + WINDOW]
        if 'X' in seq:
            continue # skip windows with unusual residues
        piece = coords[start:start + WINDOW]
        # Center at the origin: where the protein sat in the experiment's
        # coordinate frame is meaningless, only the shape matters.
        cx, cy, cz = (sum(c[k] for c in piece) / WINDOW for k in range(3))
        centered = [(x - cx, y - cy, z - cz) for x, y, z in piece]
        yield seq, centered

def write_window(f, domain_id, seq, coords):
    numbers = ' '.join(f"{v:.2f}" for xyz in coords for v in xyz)
    f.write(f"{domain_id} {seq} {numbers}\n")

counts = {'train': 0, 'test': 0}
num_domains = 0
with open('train_data.txt', 'w') as train_f, \
     open('test_data.txt', 'w') as test_f, \
     tarfile.open(ARCHIVE, 'r|gz') as archive: # r|gz: stream through it once
    for member in archive:
        if not member.isfile() or random.random() > DOMAIN_FRACTION:
            continue # not a domain file, or not in our random sample
        domain_id = os.path.basename(member.name) # e.g. "1oaiA00"
        split = 'test' if random.random() < TEST_FRACTION else 'train'
        f = test_f if split == 'test' else train_f
        lines = archive.extractfile(member).read().decode().splitlines()
        for segment in split_at_breaks(read_calphas(lines)):
            for seq, coords in windows(segment):
                write_window(f, domain_id, seq, coords)
                counts[split] += 1
        num_domains += 1
        if num_domains % 500 == 0:
            print(f"  {num_domains} domains, {counts['train'] + counts['test']} windows so far")

print(f"used {num_domains} domains: {counts['train']} train windows, "
      f"{counts['test']} test windows")
