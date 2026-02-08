# Replicating Targeted Latent Adversarial Training (LAT)

> **My first attempt at reproducing a paper**

### **Why I Chose This Paper**

Safety alignment in LLMs is often "skin deep." The paper *Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors* by Sheshadari et al. notes that standard fine-tuning often merely suppresses harmful capabilities, rather than removing them.

I chose to replicate this work because I was fascinated by the idea of achieving model robustness through pertubations in its latent space, making the model go through a sort of inner war akin to St. Augustine's concept of the "divided will." By injecting pertubations that force the model to output harmful content, then fine tuning the model to be safe anyways, we make the model's inner structures incapable of producing harmful output. 

The original paper can be found [here](https://arxiv.org/abs/2407.15549). 

### **Technical Implementation**

My implementation involved building the core LAT loop from scratch. The bulk of the implementation can be found in the notebooks folder in 02_replication.ipynb. 

1. **Latent Hook:** Created a `LATHook` class to intercept the residual stream at layers 8, 16, 24, and 30.
2. **The Inner Loop (The Attack):** Implemented a 5-step **Projected Gradient Descent (PGD)** loop. The adversary optimizes a perturbation within a sphere that forces the model to output harmful output. 
3. **The Outer Loop (The Defense):** The model weights are updated through supervised fine-tuning to minimize refusal loss.
4. **Utility Regularization:** Used **KL-Divergence** with a frozen teacher model to preserve original reasoning capabilities.


### **Results**

Below is the loss trajectory captured during my training run. The spikes in the **Total Loss** are a known feature of LAT, which represent the moments where the PGD attack found a latent weak point, which the model then learned to defend.


<img width="1373" height="545" alt="replication" src="https://github.com/user-attachments/assets/1f10daf7-1c62-4e23-904b-3722daa6a27f" />
<p align="center"><b>Figure 1:</b> W&B logs showing the adversarial "inner war." Spikes denote successful PGD-5 attacks forcing the model to re-align its latent representations.</p>
<img width="914" height="450" alt="leetspeak" src="https://github.com/user-attachments/assets/287cdf7c-760a-4806-b413-6c7162303181" />
<p align="center"><b>Figure 2:</b> A negative result. There exist jailbreaking techniques that still bypass the refusal boundary.</p>

### **Engineering Considerations**

In order to satisfy the compute requirements of training a large model, I did the following: 

* **Hardware:** Provisioned an **NVIDIA H100** via **RunPod** to run **Llama-3-8B** in `bfloat16`.
* **Optimization:** Utilized a high-rank **LoRA** configuration to ensure the model had the expressivity to learn new refusal boundaries.
* **Infrastructure:** Monitored the 10,000-step training run using **Weights & Biases** to track adversarial loss spikes.

### **What I Learned**

* **Hyperparameter Fragility:** The perturbation bound  is a double-edged sword; a bound too large destroys model utility, while too little fails to effect the weights.
* **Negative Result:** My audit showed that while the model is robust to direct harmful requests, it remained susceptible to more complex jailbreaks, suggesting that the training was not enough capture all semantic directions of attack.


