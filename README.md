# ESM3-Hb: ESM3-Driven Hybrid Design of a Hemoglobin Variant

## Project Overview

This repository contains code, data, and results for the computational design of a novel hemoglobin (Hb) variant using ESM3, a multimodal generative protein language model developed by EvolutionaryScale.

The project leverages ESM3's ability to simultaneously reason over protein sequence, structure, and function to create a hybrid hemoglobin variant with enhanced oxygen binding functionality and improved structural stability. This "hybrid" approach combines elements from natural hemoglobin sequences with AI-generated modifications guided by functional prompts.

Hemoglobin is the critical oxygen-transport protein in red blood cells. By designing variants with tailored properties, this work explores potential applications in biotechnology, therapeutics (e.g., blood substitutes or treatments for hemoglobinopathies), and fundamental protein engineering.

## Key Features

- AI-Driven Design: Uses ESM3 to generate novel hemoglobin sequences conditioned on structural templates (e.g., human Hb PDB structures) and functional keywords (e.g., "high oxygen affinity", "enhanced stability").
- In Silico Evaluation Pipeline:
  - Molecular Dynamics (MD) simulations for structural stability assessment.
  - Density Functional Theory (DFT) calculations for electronic properties and heme interactions.
  - Binding site analysis and molecular docking for ligand (O2, CO2) interactions.
- Data and Results: Generated sequences, predicted structures, simulation trajectories, and analysis outputs.

## Repository Structure

- ESM3/: Scripts for ESM3 inference, prompt engineering, and sequence generation.
- MD/: Molecular dynamics setups, trajectories, and analysis (e.g., RMSD, RMSF, free energy calculations).
- DFT/: Quantum chemistry inputs/outputs for heme environment and oxygen binding energetics.
- binding_site/: Tools for identifying and characterizing active/binding sites.
- docking/: Molecular docking results for oxygen and other ligands.
- data/: Raw and processed data, including generated Hb variants, PDB files, and summary tables.

## Requirements

- Python 3.10+
- Libraries: See requirements.txt (if available) or install via:
  pip install torch numpy pandas matplotlib biopython rdkit pyscf openpyxl
- Access to ESM3 model (via EvolutionaryScale API or open weights where available).
- Simulation tools: GROMACS/Amber for MD, Gaussian/ORCA for DFT, AutoDock Vina for docking.

## Usage

1. Generate Variants:
   cd ESM3
   python generate_hb_variants.py --prompt "hemoglobin high oxygen affinity stable tetramer"

2. Run Simulations:
   Follow scripts in respective directories for MD, DFT, and docking.

3. Analyze Results:
   Use provided notebooks or scripts in data/ for visualization and comparison to wild-type hemoglobin.

## Results Highlights

- Designed variant(s) show improved tetrameric stability in MD simulations.
- Enhanced oxygen binding affinity predicted via docking and DFT.
- Structural integrity maintained with low RMSD relative to native Hb.

Detailed comparisons and figures are in the data/ folder.

## Acknowledgments

- ESM3 model by EvolutionaryScale[](https://www.evolutionaryscale.ai/).
- Inspired by advances in generative protein design for therapeutic proteins.

## License

This project is for research and educational purposes. See LICENSE file for details.

For questions or collaborations, open an issue or contact the repository owner.
