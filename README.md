# EMDAG 
**Energy-Matched Diffusion for Antibody Design**

![cover-large](./assets/emdag.png)

---

## Installation

### Environment Setup

Create and activate the conda environment:

```bash
conda env create -f env.yaml -n emdag
conda activate emdag
```

**Note:** Configure your toolkit version in [`env.yaml`](./env.yaml) before installation.

### Datasets and Trained Weights

**SAbDab Dataset:**
- Download protein structures from the [SAbDab archive](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/)
- Extract `all_structures.zip` into the `data` folder
- Use the .tsv in the `data` folder

**Model Weights:**  
Trained model weights are made available on request.


### PyRosetta

PyRosetta is required for relaxing generated structures and computing binding energy.

- Follow the [installation instructions](https://www.pyrosetta.org/downloads)

### Ray

Ray is required for relaxation and evaluation of generated antibodies.

```bash
pip install -U ray
```

---

## Training

Train EMDAG with example configuration:

```bash
python train.py ./configs/train/emdag_train.yml
```
Adjust em_target_ratio in emdag/modules/diffusion/emdag.py to test out different energy matching strengths

---

## Sampling

Generate antibodies using the trained model (example for Kx = 1, Ko = 1):

1. Configure the checkpoint file path in `./configs/test/emdag_test.yml`
2. Configure the bash file (`run_mass_generation_emdag.sh`)
3. Run the generation script:

```bash
chmod +x run_mass_generation_emdag.sh
./run_mass_generation_emdag.sh
```

---

## Evaluation

Evaluate the generated antibodies:

```bash
python -m emdag.tools.eval
```

