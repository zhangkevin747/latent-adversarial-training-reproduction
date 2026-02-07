# Replicating Targeted Latent Adversarial Training (LAT)

> **My first attempt at reproducing a paper**

### **Why I Chose This Paper**

Safety alignment in LLMs is often "skin deep." The paper *Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors* notes that standard fine-tuning often merely suppresses harmful capabilities rather than removing them.

I chose to replicate this work because I was fascinated by the idea of making a model robust through pertubations in its latent 

### **The Stack: Engineering on the Frontier**

To achieve a high-fidelity replication of a **TMLR** paper, I moved beyond consumer hardware:

* **Hardware:** Provisioned an **NVIDIA H100** via **RunPod** to handle **Llama-3-8B** in native `bfloat16`.
* **Optimization:** Utilized a high-rank **LoRA ()** configuration to ensure the model had the expressivity to learn new refusal boundaries.
* **Infrastructure:** Monitored the 10,000-step training run using **Weights & Biases** to track adversarial loss spikes.
<img width="2461" height="545" alt="wb" src="https://github.com/user-attachments/assets/bd42075a-f18e-450f-a484-48ddc1f5de44" />

### **Technical Deep-Dive**

My implementation involved building the core LAT defense loop from scratch:

1. **Latent Hooking:** Created a `LATHook` class to intercept the residual stream at layers 8, 16, 24, and 30.
2. 
**The Inner Loop (The Attack):** Implemented a 5-step **Projected Gradient Descent (PGD)** loop. The adversary optimizes a perturbation  within an  sphere () to elicit harmful text.


3. 
**The Outer Loop (The Defense):** The model weights are updated to minimize refusal loss while  is applied.


4. 
**Utility Regularization:** Implemented a **KL-Divergence penalty** against a frozen teacher model to preserve original reasoning capabilities.



### **Observability & Results**

Below is the training trajectory captured during my H100 run. The "spikes" in the **Total Loss** are a known feature of LAT—they represent the moments where the PGD attack found a latent "weak point," which the model then learned to defend.

### **Post-Mortem: What I Learned**

* **Hyperparameter Fragility:** The perturbation bound  is a double-edged sword; too much power destroys model utility, while too little fails to "stress" the weights.
* **The Leetspeak Gap:** My audit showed that while the model became robust to direct harmful requests, it remained susceptible to complex **Leetspeak jailbreaks**, suggesting that  perturbation balls don't yet capture all semantic directions of attack.
