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
Figure 1 illustrates that our VABAM model excels at amplitude-based modulation, maintaining the original morphology of signals by avoiding horizontal shifts in both phase and time axes. Conversely, conditional VAEs struggle to manage these shifts when PSD values are introduced as additional input.
<br><br>


- **Introduction of Novel Metrics:** We propose three novel metrics to provide a comprehensive evaluation of the model's synthesis and representation capabilities:
  1. **Disentanglement of Z in Signal Morphology:** Assessing the model's ability to separate different aspects of the signal morphology.
  2. **Contribution of Ancillary Information to Signal Morphology:** Evaluating how additional (conditional) information affects the signal morphology.
  3. **Controllability of Amplitude-based Synthesis within Morphological Identities:** Measuring the model's capability to modulate signal amplitude without altering its morphological identity.
 

## Research Motivation
<p align="center">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/signal_filter_animation.gif" width="49%" alt="Pass-filter mechanism">
  <img src="https://github.com/JunetaeKim/VABAM/blob/main/Figures/processing_animation.gif" width="49%" alt="Trade-off between cleansing efficacy and morphological alteration in pulsatile signals">
  <br>
  <em> Figure 2: Pass-filter mechanism (left) vs Trade-off between cleansing efficacy and morphological alteration in pulsatile signals (right) </em>  
</p>


