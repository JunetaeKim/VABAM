## VABAM: Variational Autoencoder for Amplitude-based Biosignal Augmentation within Morphological Identities.

This repository contains the Python code for VABAM and the Joint Mutual Information (JMI) based metrics introduced in our paper. Our research focuses on the synthesis of pulsatile physiological signals, emphasizing the modulation of amplitude while preserving the signals' morphological identities. Please access **our working paper** [here](https://www.techrxiv.org/users/146056/articles/737765-vabam-variational-autoencoder-for-amplitude-based-biosignal-augmentation-within-morphological-identities).

### Research Highlights

- **Development of the VABAM Model:** A model capable of synthesizing pulsatile physiological signals through pass filter effects, namely *amplitude-based* modulation, ensuring the preservation of the signals' morphological identity.
<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Anim.%201%20VABAM%20(Our%20Model)%20Synthesis%20Results.gif" width="49%" alt="Pass-filter mechanism">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Anim.%202%20C-VAE%20Synthesis%20Results.gif" width="49%" alt="Pass-filter mechanism">
  <br>
  <em>Figure 1: Amplitude-Based Modulation of ABP via VABAM (left) vs CVAE (right) </em>  
</p>
Figure 1 shows the results of synthesizing 100 signals from a single original Arterial Blood Pressure (ABP). VABAM excels in maintaining the original morphology of signals during synthesis by avoiding phase alterations and horizontal shifts in the time axes. Conversely, conditional VAEs struggle to maintain morphological identities when PSD values are incorporated as conditional input.
<br><br>


- **Introduction of Novel Metrics:** We propose three novel metrics to provide a comprehensive evaluation of the model's synthesis and representation capabilities:
  1. **Disentanglement of Z in Signal Morphology:** Assessing the model's ability to separate different aspects of the signal morphology.
  2. **Contribution of Ancillary Information to Signal Morphology:** Evaluating how additional (conditional) information affects the signal morphology.
  3. **Controllability of Amplitude-based Synthesis within Morphological Identities:** Measuring the model's capability to modulate signal amplitude without altering its morphological identity.
 

### Research Motivation
- **Challenges in Signal Processing:** Signals often face noise and artifacts from various sources like electrical interference, motion, and external physiological factors, complicating accurate diagnostic information extraction. Using frequency pass filters to mitigate these problems can lead to edge effects, causing signal truncation that may compromise crucial information at the signal boundaries (Figure 2). Furthermore, these filters may also result in phase alterations and horizontal shifts along the time axis (Figure 2).
<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Anim.%203%20Pass%20Filter%20Convolution%20Operation%20Animation.gif" width="49%" alt="Pass-filter mechanism">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Anim.%204%20Trade-off%20between%20Cleansing%20Efficacy%20and%20Morphological%20Alteration.gif" width="49%" alt="Trade-off between cleansing efficacy and morphological alteration in pulsatile signals">
  <br>
  <em> Figure 2: Pass-filter mechanism (left) vs Trade-off between cleansing efficacy and morphological alteration (right) </em>  
</p>

- **Enhancement of Artificial Intelligence Models Through Preservation of Signal Morphology in Synthesis:** The ability to maintain the original shape of signals during synthesis can significantly enhance the capabilities of rapidly evolving artificial intelligence models that utilize frequency and amplitude-based features of physiological signals, as shown through an example in [our previous work](https://ieeexplore.ieee.org/document/10130807).
<p align="center">
  <img src="https://github.com/JunetaeKim/DWT-HPI/blob/main/Figures/ScenarioBasedGuideline.jpg" width="60%" alt="HPI Model">
  <br>
  <em> Figure 3: Application for Predicting Hypotension Utilizing Amplitude-Based Features </em>  
</p><br><br>

## A Brief Introduction to VABAM
-VABAM is structured around five key components: Feature Extractor, Encoder, Sampler, Feature Generator, and Signal Reconstructor (Figure 4). For detailed information, please refer to our paper.

- **Feature Extractor** $\boldsymbol{g_{x}(\cdot)}$ applies cascading filters to the raw signal $y$, producing four amplitude-modulated subsets $x \in \{x_{HH}, x_{HL}, x_{LH}, x_{LL}\}$ that guide the Feature Generator.

- **Encoder** $\boldsymbol{g_{e}(\cdot)}$ learns parameters for the latent variable $Z$ and cutoff frequency $\Theta$, under two assumptions:
  - $\theta_k \sim \mathcal{U}(0, 1)$ for $k = 1, \ldots, 6$, indicating six instances in the model, approximated by a Bernoulli distribution.
  - $z_{j} \sim \mathcal{N}(\mu_{z_j}, \sigma_{z_j}^2)$ for each dimension $j$, with $j \in \{1, 2, \ldots, J\}$, where $J$ is a hyperparameter defining dimension count.

- **Sampler** $\boldsymbol{g_{z}(\cdot)}$ and $\boldsymbol{g_{\theta}(\cdot)}$ utilizes the reparameterization trick for backpropagation, allowing sampling of $z_{j}$ and $\theta_{k}$ for gradient flow.

- **Feature Generator** $\boldsymbol{g_{x'}(\cdot)}$ generates four principal feature signals for the Signal Reconstructor, aligning with the amplitude-modulated subsets from the Feature Extractor.

- **Signal Reconstructor** $\boldsymbol{g_{y}(\cdot)}$ reconstructs coherent signals from the feature subsets, keeping the original signal's main aspects and adding latent elements influenced by $z_{j}$ and $\theta_{k}$.

<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Training%20and%20Generating%20Framework.png" width="60%" alt="Intuitive Illustration of VABAM">
  <br>
  <em> Figure 4: Intuitive Illustration of VABAM </em>  
</p><br><br>

## Library Dependencies and Test Environment Information
VABAM's training and its post-evaluation were conducted and tested with the following libraries and their respective versions:
- Python == 3.8.16 , 3.9.18
- numpy == 1.19.5 , 1.26.0
- pandas == 1.1.4 , 2.1.1
- tensorflow == 2.4.0 , 2.10.0
- gpu == rtx4080 , rtx4090
<br><br>

## Code Overview and Run Procedure Guide
### For Training
To start the training process, use the following scripts:
- `TrainModel.py`: Script for training the main model.
- `TrainBenchmark.py`: Script for training benchmark models. Refer to the [Benchmarks](https://github.com/JunetaeKim/VABAM/tree/main/Benchmarks) folder.

### For JMI-Based Metric Computation
To compute the JMI-based metrics, follow these steps:
- `SubProcMIEVAL.py` (with `BatchMIEvaluation.py`): Script for computing metrics.
- `SubProcMIEVAL.py` (with `BatchBMMIEvaluation.py`): Script for computing benchmark model metrics. Refer to the [Benchmarks](https://github.com/JunetaeKim/VABAM/tree/main/Benchmarks) folder.
- `TabulatingResults.py`: Script for tabulating results from the main model evaluation.
- `TabulatingBMResults.py`: Script for tabulating results from the benchmark model evaluation. Refer to the [Benchmarks](https://github.com/JunetaeKim/VABAM/tree/main/Benchmarks) folder.
For visualization and table generation:
- `VisualizationSig.ipynb`: Jupyter notebook for signal visualization.
- `VisualizationMetrics.ipynb`: Jupyter notebook for metrics visualization.
- `Tables.ipynb`: Jupyter notebook for generating tables of results.
- Please note that the visualization code heavily relies on GPT-4.0 and was not primarily written with high readability in mind.This is an ongoing coding file, so the code may contain redundancies and is subject to continuous updates.

### For Fine-Tuning (Optional)
For optional fine-tuning of the model, use:
- `FineTuneModel.py`: Script for fine-tuning the model based on specific needs or data.
- `FineTuneBenchmark.py`: Script for fine-tuning benchmark models. Refer to the [Benchmarks](https://github.com/JunetaeKim/VABAM/tree/main/Benchmarks) folder.

Please consult the documentation within each script for more detailed instructions on usage and parameters.

### Configurations
- Configuration files for the main and benchmark models are located in the [Config](https://github.com/JunetaeKim/VABAM/tree/main/Config) and [/Benchmarks
/Config/](https://github.com/JunetaeKim/VABAM/tree/main/Benchmarks/Config) folders, respectively.
<br><br>

## Scripts Executed for Our Research
All execution code lists are available in the [ExecutionProcedure.txt](https://github.com/JunetaeKim/VABAM/blob/main/ExecutionProcedure.txt) file; please refer to this file for detailed information.

### 1.Dataset
You can download the processed dataset by running GitBash or Command Prompt and using wget, or you can directly download it via the URL.

**Download link:** https://www.dropbox.com/scl/fi/g6f83ooxtg5p3bz4m6aur/ProcessedData.egg?rlkey=bb74m27fyqm4e73s960aeq6z0&dl=1



### 2.TrainModel.py 

**MainModel Training Commands:**
python TrainModel.py --Config [model_config] --GPUID [gpu_id]

- **ConfigART500 Examples:**
  - `python .\TrainModel.py --Config SKZFC_ART_30_500 --GPUID 0`
  - `python .\TrainModel.py --Config SKZFC_ART_50_500 --GPUID 0`
  - `python .\TrainModel.py --Config FACFC_ART_30_500 --GPUID 0`
  - `python .\TrainModel.py --Config FACFC_ART_50_500 --GPUID 0`
  - `python .\TrainModel.py --Config SKZ_ART_30_500 --GPUID 0`
  - `python .\TrainModel.py --Config SKZ_ART_50_500 --GPUID 0`
  - `python .\TrainModel.py --Config TCMIDKZFC_ART_30_500 --GPUID 0`
  - `python .\TrainModel.py --Config TCMIDKZFC_ART_50_500 --GPUID 0`

- **ConfigART800 Examples:**
  - `python .\TrainModel.py --Config SKZFC_ART_30_800 --GPUID 0`
  - ...
  - `python .\TrainModel.py --Config TCMIDKZFC_ART_50_800 --GPUID 0`

- **ConfigII500 Examples:**
  - `python .\TrainModel.py --Config SKZFC_II_30_500 --GPUID 0`
  - ...
  - `python .\TrainModel.py --Config TCMIDKZFC_II_50_500 --GPUID 0`

- **ConfigII800 Examples:**
  - `python .\TrainModel.py --Config SKZFC_II_30_800 --GPUID 0`
  - ...
  - `python .\TrainModel.py --Config TCMIDKZFC_II_50_800 --GPUID 0`
<br>

**Benchmark Model Training Commands:**
python TrainBenchmark.py --Config [model_config] --GPUID [gpu_id]

- **ConfigART Examples:**
  - `python .\TrainBenchmark.py --Config TCVAE_ART_30 --GPUID 0`
  - ...
  - `python .\TrainBenchmark.py --Config BaseVAE_ART_50 --GPUID 0`

- **ConfigII Examples:**
  - `python .\TrainBenchmark.py --Config TCVAE_II_30 --GPUID 0`
  - ...
  - `python .\TrainBenchmark.py --Config BaseVAE_II_50 --GPUID 0`
<br><br>

### 3. SubProcMIEVAL.py
**MainModel Training Commands:**
python SubProcMIEVAL.py --Config [eval_config] --GPUID [gpu_id] --ConfigSpec [model_spec] --SpecNZs [nz_values] --SpecFCs [fc_values]

- **MainModel Examples:**
  - `python .\SubProcMIEVAL.py --Config EvalConfigART800 --GPUID 4`
  - ...
  - `python .\SubProcMIEVAL.py --Config EvalConfigII500 --GPUID 4`
<br>

**Benchmark Model Training Commands:**
python SubProcMIEVAL.py --Config [eval_config] --GPUID [gpu_id]

- **Benchmark Examples:**
  - `python .\SubProcMIEVAL.py --Config EvalConfigART --GPUID 4`
  - `python .\SubProcMIEVAL.py --Config EvalConfigII --GPUID 4`
<br><br>

### 4. TabulatingResults.py
**MainModel Training Commands:**
python TabulatingResults.py -CP [config_path] --GPUID [gpu_id]

- **MainModel Example:**
  - `python .\TabulatingResults.py -CP ./Config/ --GPUID 4`
<br>

**Benchmark Model Training Commands:**
python TabulatingBMResults.py -CP [config_path] --GPUID [gpu_id]

- **Benchmark Example:**
  - `python .\TabulatingBMResults.py -CP ./Config/ --GPUID 4`


