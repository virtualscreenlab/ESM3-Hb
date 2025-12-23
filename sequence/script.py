from Bio import pairwise2
from Bio.pairwise2 import format_alignment

# Define the sequences
seq1 = (
    "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAV"
    "AHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
)

seq2 = (
    "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLT"
    "ALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRK"
    "DIAAKYKELGYQG"
)

# Perform a global alignment
alignments = pairwise2.align.globalxx(seq1, seq2)

# Display the first alignment (best alignment)
print(format_alignment(*alignments[0]))

# Extract the aligned sequences
aligned_seq1 = alignments[0][0]
aligned_seq2 = alignments[0][1]

# Calculate sequence identity
matches = sum(c1 == c2 for c1, c2 in zip(aligned_seq1, aligned_seq2))
alignment_length = max(len(aligned_seq1), len(aligned_seq2))
sequence_identity = (matches / alignment_length) * 100

# Calculate sequence similarity (excluding gaps)
similar_matches = sum(c1 == c2 for c1, c2 in zip(aligned_seq1, aligned_seq2) if c1 != '-' and c2 != '-')
similarity_length = len([c for c in aligned_seq1 if c != '-'])
sequence_similarity = (similar_matches / similarity_length) * 100

# Print results
print(f"Sequence Identity: {sequence_identity:.2f}%")
print(f"Sequence Similarity (excluding gaps): {sequence_similarity:.2f}%")

