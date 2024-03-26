## VABAM: Variational Autoencoder for Amplitude-based Biosignal Augmentation within Morphological Identities.

This repository contains the Python code for VABAM and the Joint Mutual Information (JMI) based metrics introduced in our paper. Our research focuses on the synthesis of pulsatile physiological signals, emphasizing the modulation of amplitude while preserving the signals' morphological identities.

## Research Highlights

- **Development of the VABAM Model:** A model capable of synthesizing pulsatile physiological signals through pass filter effects, namely *amplitude-based* modulation, ensuring the preservation of the signals' morphological identity.
<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/diffusion_animation_vabam_gradient_color55.gif" width="49%" alt="Pass-filter mechanism">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/diffusion_animation_convae_gradient_color55.gif" width="49%" alt="Pass-filter mechanism">
  <br>
  <em>Figure 1: Amplitude-Based Modulation of ABP via VABAM (left) vs CVAE (right) </em>  
</p>
Figure 1 illustrates that VABAM excels in maintaining the original morphology of signals during synthesis by avoiding horizontal shifts in both phase and time axes. Conversely, conditional VAEs struggle to maintain morphological identities when PSD values are incorporated as conditional input.
<br><br>


- **Introduction of Novel Metrics:** We propose three novel metrics to provide a comprehensive evaluation of the model's synthesis and representation capabilities:
  1. **Disentanglement of Z in Signal Morphology:** Assessing the model's ability to separate different aspects of the signal morphology.
  2. **Contribution of Ancillary Information to Signal Morphology:** Evaluating how additional (conditional) information affects the signal morphology.
  3. **Controllability of Amplitude-based Synthesis within Morphological Identities:** Measuring the model's capability to modulate signal amplitude without altering its morphological identity.
 

## Research Motivation
- **Challenges in Signal Processing:** Signals often face noise and artifacts from various sources like electrical interference, motion, and external physiological factors, complicating accurate diagnostic information extraction. Using frequency pass filters to mitigate these problems can lead to edge effects, causing signal truncation that may compromise crucial information at the signal boundaries (Figure 2). Furthermore, these filters may also result in phase alterations and horizontal shifts along the time axis (Figure 2).
<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/signal_filter_animation.gif" width="49%" alt="Pass-filter mechanism">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/processing_animation.gif" width="49%" alt="Trade-off between cleansing efficacy and morphological alteration in pulsatile signals">
  <br>
  <em> Figure 2: Pass-filter mechanism (left) vs Trade-off between cleansing efficacy and morphological alteration (right) </em>  
</p>

- **Enhancement of Artificial Intelligence Models Through Preservation of Signal Morphology in Synthesis:** The ability to maintain the original shape of signals during synthesis can significantly enhance the capabilities of rapidly evolving artificial intelligence models that utilize frequency and amplitude-based features of physiological signals, as shown through an example in [our previous work](https://ieeexplore.ieee.org/document/10130807).
<p align="center">
  <img src="https://github.com/JunetaeKim/DWT-HPI/blob/main/Figures/ScenarioBasedGuideline.jpg" width="60%" alt="HPI Model">
  <br>
  <em> Figure 3: Application for Predicting Hypotension Utilizing Amplitude-Based Features </em>  
</p>

## A Brief Introduction to VABAM
-VABAM is structured around five key components: Feature Extractor, Encoder, Sampler, Feature Generator, and Signal Reconstructor (Figure 4). 

- **Feature Extractor** (\(\boldsymbol{g_{x}(\cdot)}\)) applies cascading filters to the raw signal \(y\), producing four amplitude-modulated subsets \(x \in \{x_{HH}, x_{HL}, x_{LH}, x_{LL}\}\) that guide the Feature Generator.

- **Encoder** (\(\boldsymbol{g_{e}(\cdot)}\)) learns parameters for the latent variable \textcolor{red}{\(Z\)} and cutoff frequency \textcolor{red}{\(\Theta\)}, under two assumptions:
  - \( \theta_k \sim \mathcal{U}(0, 1) \) for \( k = 1, \ldots, 6 \), indicating six instances in the model, approximated by a Bernoulli distribution (refer to Eq.(\ref{Approximated Bern})).
  - \(z_{j} \sim \mathcal{N}(\mu_{z_j}, \sigma_{z_j}^2)\) for each dimension \(j\), with \(j \in \{1, 2, \ldots, J\}\), where \(J\) is a hyperparameter defining dimension count.

- **Sampler** (\(\boldsymbol{g_{z}(\cdot)}\) and \(\boldsymbol{g_{\theta}(\cdot)}\)) utilizes the reparameterization trick for backpropagation, allowing sampling of $z_{j}$ and $\theta_{k}$ for gradient flow.

- **Feature Generator** (\(\boldsymbol{g_{x'}(\cdot)}\)), with Encoder-derived parameters, generates four principal feature signals for the Signal Reconstructor, aligning with the amplitude-modulated subsets from the Feature Extractor.

- **Signal Reconstructor** (\(\boldsymbol{g_{y}(\cdot)}\) reconstructs coherent signals from the feature subsets, keeping the original signal's main aspects and adding latent elements influenced by $z_{j}$ and $\theta_{k}$.

<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/Training%20and%20Generating%20Framework.png" width="60%" alt="Intuitive Illustration of VABAM">
  <br>
  <em> Figure 4: Intuitive Illustration of VABAM </em>  
</p>
