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

- **Enhancement of Artificial Intelligence Models Through Preservation of Signal Morphology in Synthesis:** The ability to maintain the original shape of signals during synthesis can significantly enhance the capabilities of rapidly evolving artificial intelligence models that utilize frequency and amplitude-based features of physiological signals.
<p align="center">
  <img src="https://github.com/JunetaeKim/DWT-HPI/blob/main/Figures/ScenarioBasedGuideline.jpg" width="49%" alt="HPI Model">
  <br>
  <em> Figure 3: Application for Predicting Hypotension Utilizing Amplitude-Based Features </em>  
</p>

