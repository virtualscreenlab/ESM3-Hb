from IPython.display import clear_output

#!pip install git+https://github.com/evolutionaryscale/esm.git
#!pip install py3Dmol

clear_output()  # Suppress pip install log lines after installation is complete.

from getpass import getpass

import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import matplotlib.pyplot as pl
import py3Dmol
import torch
from esm.sdk import client
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

token = "XXX"

model = client(
    model="esm3-medium-2024-03", url="https://forge.evolutionaryscale.ai", token=token
)

template_gfp = ESMProtein.from_protein_chain(
    ProteinChain.from_rcsb("1a3n", chain_id="A")
)

template_gfp_tokens = model.encode(template_gfp)

print("Sequence tokens:")
print(
    "    ", ", ".join([str(token) for token in template_gfp_tokens.sequence.tolist()])
)

print("Structure tokens:")
print(
    "    ", ", ".join([str(token) for token in template_gfp_tokens.structure.tolist()])
)

prompt_sequence = ["_"] * len(template_gfp.sequence)
prompt_sequence[44] = "R"
prompt_sequence = "".join(prompt_sequence)

print(template_gfp.sequence)
print(prompt_sequence)

prompt = model.encode(ESMProtein(sequence=prompt_sequence))

# We construct an empty structure track like |<bos> <mask> ... <mask> <eos>|...
prompt.structure = torch.full_like(prompt.sequence, 4096)
prompt.structure[0] = 4098
prompt.structure[-1] = 4097

prompt.structure[0:100] = template_gfp_tokens.structure[0:100]
#prompt.structure[0:90] = template_gfp_tokens.structure[0:90]
#prompt.structure[0:80] = template_gfp_tokens.structure[0:80]
#prompt.structure[0:70] = template_gfp_tokens.structure[0:70]
#prompt.structure[0:60] = template_gfp_tokens.structure[0:60]
#prompt.structure[0:50] = template_gfp_tokens.structure[0:50]
#prompt.structure[0:40] = template_gfp_tokens.structure[0:40]
#prompt.structure[0:30] = template_gfp_tokens.structure[0:30]
#prompt.structure[0:20] = template_gfp_tokens.structure[0:20]
#prompt.structure[0:15] = template_gfp_tokens.structure[0:15]
#prompt.structure[0:10] = template_gfp_tokens.structure[0:10]
#prompt.structure[0] = template_gfp_tokens.structure[0]

print("".join(["✔" if st < 4096 else "_" for st in prompt.structure]))

num_tokens_to_decode = (prompt.structure == 4096).sum().item()

structure_generation = model.generate(
    prompt,
    GenerationConfig(
        # Generate a structure.
        track="structure",
        # Sample one token per forward pass of the model.
        num_steps=num_tokens_to_decode,
        # Sampling temperature trades perplexity with diversity.
        temperature=1.0,
    ),
)

print("These are the structure tokens corresponding to our new design:")
print(
    "    ", ", ".join([str(token) for token in structure_generation.structure.tolist()])
)

# Decodes structure tokens to backbone coordinates.
structure_generation_protein = model.decode(structure_generation)

print("")

view = py3Dmol.view(width=1000, height=500)
view.addModel(
    structure_generation_protein.to_protein_chain().infer_oxygen().to_pdb_string(),
    "pdb",
)
view.setStyle({"cartoon": {"color": "lightgreen"}})
view.zoomTo()
view.show()

constrained_site_positions = [44]

template_chain = template_protein.to_protein_chain()
generation_chain = structure_generation_protein.to_protein_chain()

constrained_site_rmsd = template_chain[constrained_site_positions].rmsd(
    generation_chain[constrained_site_positions]
)
backbone_rmsd = template_chain.rmsd(generation_chain)

c_pass = "✅" if constrained_site_rmsd < 1.5 else "❌"
b_pass = "✅" if backbone_rmsd > 1.5 else "❌"

print(f"Constrained site RMSD: {constrained_site_rmsd:.2f} Ang {c_pass}")
print(f"Backbone RMSD: {backbone_rmsd:.2f} Ang {b_pass}")


num_tokens_to_decode = (prompt.sequence == 32).sum().item()

sequence_generation = model.generate(
    # Generate a sequence.
    structure_generation,
    GenerationConfig(track="sequence", num_steps=num_tokens_to_decode, temperature=1.0),
)

# Refold
sequence_generation.structure = None
length_of_sequence = sequence_generation.sequence.numel() - 2
sequence_generation = model.generate(
    sequence_generation,
    GenerationConfig(track="structure", num_steps=length_of_sequence, temperature=0.0),
)

# Decode to AA string and coordinates.
sequence_generation_protein = model.decode(sequence_generation)

sequence_generation_protein.sequence

import matplotlib.pyplot as pl

seq1 = seq.ProteinSequence(template_protein.sequence)
seq2 = seq.ProteinSequence(sequence_generation_protein.sequence)

alignments = align.align_optimal(
    seq1, seq2, align.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1)
)

alignment = alignments[0]

identity = align.get_sequence_identity(alignment)
print(f"Sequence identity: {100*identity:.2f}%")

print("\nSequence alignment:")
fig = pl.figure(figsize=(8.0, 4.0))
ax = fig.add_subplot(111)
graphics.plot_alignment_similarity_based(
    ax, alignment, symbols_per_line=45, spacing=2, show_numbers=True
)
fig.tight_layout()
pl.show()

template_chain = template_protein.to_protein_chain()
generation_chain = sequence_generation_protein.to_protein_chain()

constrained_site_rmsd = template_chain[constrained_site_positions].rmsd(
    generation_chain[constrained_site_positions]
)
backbone_rmsd = template_chain.rmsd(generation_chain)

c_pass = "✅" if constrained_site_rmsd < 1.5 else "❌"
b_pass = "🤷‍♂️"

print(f"Constrained site RMSD: {constrained_site_rmsd:.2f} Ang {c_pass}")
print(f"Backbone RMSD: {backbone_rmsd:.2f} Ang {b_pass}")

view = py3Dmol.view(width=600, height=600)
view.addModel(sequence_generation_protein.to_pdb_string(), "pdb")
view.setStyle({"cartoon": {"color": "lightgreen"}})
view.zoomTo()
view.show()

# **9. Saving the Final Structure**
with open("final_structure.pdb", "w") as f:
    f.write(sequence_generation_protein.to_pdb_string())

print("Final mutated structure saved to final_structure.pdb")


print(seq1)
print(seq2)