# Active Learning Identifies Sulfur-Based Enhancers for Fe(III)-Protoporphyrin Catalysis

This repository accompanies the manuscript:

**Active Learning Identifies Sulfur-Based Enhancers for Fe(III)-Protoporphyrin Catalysis: Recapitulating Features of Natural Oxidase and Beyond**

The study builds a peptide-based artificial oxidase platform by covalently anchoring Fe(III)-protoporphyrin IX to a lysine side chain in synthetic decapeptides. Using hydrogen peroxide as the oxidant and acetophenone as the model substrate, the project combines peptide synthesis, catalytic screening, and active learning to identify sequence features that improve turnover number (TON).

## What the paper shows

- Active learning was used to explore peptide sequence space over 20 optimization rounds.
- The manuscript reports 233 experimentally tested peptide variants in the closed-loop campaign.
- Catalytic performance improved over the campaign, reaching a best TON of 26.17.
- Sulfur-containing residues, especially cysteine and methionine near the heme-anchoring lysine, were strongly associated with higher activity.
- Cysteine recapitulates a key motif found in natural heme-thiolate oxidases, while methionine emerged as an additional sulfur-based enhancer not typically emphasized in natural oxidase design.
- Spectroscopy and DFT analysis support sulfur coordination as a mechanistic explanation for the observed rate enhancement.

## Repository layout

| Path | Purpose |
| --- | --- |
| `Active_Learning/MAP_Elites.ipynb` | Early-stage quality-diversity search corresponding to the exploratory phase of the campaign. |
| `Active_Learning/Surrogate_AL.ipynb` | Later-stage surrogate-model workflow using sequence embeddings for iterative candidate ranking and selection. |
| `AL_vs_Random/Surrogate_AL.ipynb` | Comparison notebook for active learning versus random selection baselines. |
| `AL_vs_Random/random_baseline_from_models.csv` | Saved random-baseline comparison results. |
| `Machine_Learning/Machine_Learning.ipynb` | Sequence-activity analysis notebook focused on residue identity, position effects, and TON trends. |
| `Machine_Learning/Machine_Learning_hydro_CM.ipynb` | Additional analysis notebook with hydrophobicity and sulfur-residue-focused analysis. |
| `Machine_Learning/data_all.xlsx` | Main working spreadsheet used by the analysis and modeling notebooks. |
| `Machine_Learning/data_all_old.xlsx` | Older snapshot of the working spreadsheet. |
| `Machine_Learning/ESM2_TON_Generator/` | Standalone Python package for ESM-2-based TON regression, candidate generation, and conditional LoRA experiments. |

## Data snapshot

The spreadsheets in `Machine_Learning/` store peptide identifiers or sequences together with catalytic measurements such as:

- `ton`
- `para-Hydroxyacetophenone Selectivity(%)`
- `meta-Hydroxyacetophenone Selectivity(%)`
- `ortho-Hydroxyacetophenone Selectivity(%)`

The manuscript describes a 233-variant active learning campaign. The Excel files in this repository appear to be broader working datasets used for downstream analysis, so they should be treated as analysis tables rather than a strict manuscript-only export.

## How to use this repository

### Notebook-first workflow

If you want to follow the study logic from screening to interpretation, a good order is:

1. `Active_Learning/MAP_Elites.ipynb`
2. `Active_Learning/Surrogate_AL.ipynb`
3. `AL_vs_Random/Surrogate_AL.ipynb`
4. `Machine_Learning/Machine_Learning.ipynb`
5. `Machine_Learning/Machine_Learning_hydro_CM.ipynb`

### Packaged ESM-2 workflow

For a more structured modeling workflow, see [`Machine_Learning/ESM2_TON_Generator/`](Machine_Learning/ESM2_TON_Generator/). That subproject includes:

- environment files
- training scripts
- generation scripts
- evaluation scripts
- tests

Its setup and usage instructions are documented in [`Machine_Learning/ESM2_TON_Generator/README.md`](Machine_Learning/ESM2_TON_Generator/README.md).

## Relation to the manuscript workflow

The paper's experimental loop can be read through this repository as:

1. Explore sequence space broadly with `MAP-Elites`, then shift to embedding-based surrogate selection in the later optimization rounds.
2. Synthesize peptide-heme catalysts and evaluate acetophenone oxidation experimentally.
3. Update models with new TON and selectivity data.
4. Analyze residue-level trends, especially the role of sulfur-containing side chains.
5. Extend interpretation with spectroscopy and DFT discussed in the manuscript.

The experimental synthesis, HPLC evaluation, and spectroscopy are described in the manuscript, while this repository primarily captures the computational analysis and sequence-selection side of the project.

## Data and code availability

- GitHub repository: <https://github.com/Wang-Group/Fe-Protoporphyrin_Peptide_Catalysis>
- Figshare dataset DOI: <https://doi.org/10.6084/m9.figshare.30112690>

## Citation

If you use this repository, please cite the manuscript and the Figshare dataset associated with the project.

## Contact

- Yibin Jiang: <yibinjiang@xmu.edu.cn>
- Cheng Wang: <wangchengxmu@xmu.edu.cn>
