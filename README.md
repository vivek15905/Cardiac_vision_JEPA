# Cardiac_vision_JEPA



# Cardiac Vision-JEPA: Self-Supervised Latent Physics for Echocardiography

A highly optimized, fully customized Joint Embedding Predictive Architecture (JEPA) built from scratch using JAX and Flax. This project explores the use of self-supervised Vision Transformers to learn the biomechanical structure of human cardiac tissue directly from high-resolution ultrasound data.

## The Unmet Clinical Need
In modern computational medicine and autonomous surgical robotics, predictive models are bottlenecked by sensor noise. Ultrasound (echocardiography) is portable and non-invasive, but it is notoriously noisy, filled with stochastic speckle patterns, acoustic shadows, and artifact dropouts. 

Traditional generative AI (like Masked Autoencoders) attempts to detect anomalies by reconstructing missing pixels. When applied to ultrasound, these models waste massive computational power attempting to perfectly recreate random acoustic noise, leading to catastrophic false positives. To advance *in silico* medical simulations and robotic interventions, we need foundation models that understand the **underlying anatomical semantics**, completely ignoring the superficial noise.

##  Why JEPA?
This project utilizes an Image Joint-Embedding Predictive Architecture (I-JEPA). 
Instead of operating in pixel space, JEPA operates entirely in the **abstract latent space**. 
1. The model is given a "Context Block" of visible cardiac tissue.
2. It predicts the mathematical representation (embedding) of a hidden "Target Block".
3. Loss is calculated as the L2 distance between the predicted embedding and the actual target embedding.

Because the model never reconstructs pixels, it naturally learns to ignore stochastic speckle and focuses entirely on structural geometry and tissue dynamics. To prevent representation collapse, the Target Encoder's weights are strictly updated via an Exponential Moving Average (EMA) of the Context Encoder's weights.

##  The Dataset
**CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)**
This architecture was trained on the CAMUS dataset, consisting of high-quality 2D echocardiogram sequences.
* **Views:** 2-Chamber (2CH) and 4-Chamber (4CH) apical views.
* **Preprocessing:** Native 384x384 frames dynamically resized to 256x256 tensors via PyTorch transforms, normalized to `mean=0.5, std=0.5`.
* **Self-Supervised Setup:** Ground-truth segmentation masks were explicitly ignored. The model learns purely from the raw physical ultrasound frames.

##  Architecture Specifications
* **Framework:** JAX / Flax / Optax
* **Hardware:** Compiled for TPU v2 / NVIDIA T4 GPUs via XLA.
* **Encoders:** Vision Transformers (ViT) patchified into 16x16 grids.
* **Masking Strategy:** Spatially disjoint block masking (1 large contiguous Context Block, up to 4 disjoint Target Blocks).
* **Data Pipeline:** PyTorch `DataLoader` yielding asynchronous batches converted to NumPy arrays for JAX ingestion.

### Computational Graph (Vision-JEPA)
```mermaid
flowchart TB
    Input[Ultrasound Frame 256x256] --> Masking{Spatial Block Masking}

    %% Context Branch
    Masking -- "Context Blocks (Visible)" --> C_Patches[Context Patches]
    C_Patches --> C_Enc[Context Encoder ViT]
    C_Enc --> C_Embed[Context Embeddings]

    %% Target Branch
    Masking -- "Target Blocks (Hidden)" --> T_Patches[Target Patches]
    T_Patches --> T_Enc[Target Encoder ViT]
    T_Enc -- "Stop Gradient" --> T_Embed[Actual Target Embeddings]

    %% Prediction
    C_Embed --> Pred[Predictor ViT]
    Mask_Tokens[Mask Tokens + Positional Encodings] --> Pred
    Pred --> Pred_Embed[Predicted Target Embeddings]

    %% Loss & Updates
    Pred_Embed --> Loss((L2 Latent Loss))
    T_Embed --> Loss

    %% Parameter Updates
    Loss -. "Backprop (AdamW)" .-> C_Enc
    Loss -. "Backprop (AdamW)" .-> Pred
    C_Enc -. "EMA Update (Tau)" .-> T_Enc

    %% Styling
    classDef encoder fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef predictor fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef data fill:#f3e5f5,stroke:#4a148c,stroke-width:1px;
    classDef loss fill:#ffebee,stroke:#b71c1c,stroke-width:2px;

    class C_Enc,T_Enc encoder;
    class Pred predictor;
    class Input,C_Patches,T_Patches,C_Embed,T_Embed,Pred_Embed,Mask_Tokens data;
    class Loss loss;
```` ``` ````

```


##  Implementation Journey (Steps Followed)

### Phase 1: XLA Tracing & Foundation Building
* Architected the ViT Encoders and Predictor using `flax.linen`.
* Solved JAX initialization bottlenecks (`ScopeCollectionNotFound`) by strictly segregating the initialization dummy pass from the active EMA training pass.
* Replaced dynamic boolean array slicing with **Masked Zeroing** (`jnp.where`) to guarantee static tensor shapes, allowing the XLA compiler to perfectly optimize the computation graph.

### Phase 2: Hybrid Data Pipeline
* Built a custom `torch.utils.data.Dataset` to ingest the CAMUS HDF5 files.
* Bridged PyTorch's asynchronous batching with JAX's functional requirements, successfully streaming real echocardiogram arrays directly into GPU VRAM.

### Phase 3: Stabilizing the Training Dynamics
* Addressed early **representation divergence** (where target and context embeddings span out of control).
* Implemented `optax.chain` to introduce a warmup cosine decay learning rate schedule and global gradient clipping (norm=1.0), forcing the model to map the complex speckle instead of collapsing to zero.

### Phase 4: Marathon Training & Checkpointing
* Scaled the training loop to 100 epochs.
* Dynamically annealed the EMA momentum ($\tau$) from 0.996 to 1.0.
* Integrated Google **Orbax Checkpointing** to asynchronously serialize the `TrainState` and target network parameters to Google Drive to survive ephemeral cloud disconnections.

### Phase 5: Vectorized Inference & Diagnostic Reality
* Injected simulated structural anomalies (localized acoustic shadows) into clean test frames.
* Built a high-performance inference engine using `jax.vmap` to vectorize the latent prediction across all 256 patches simultaneously.
* **Engineering Result:** The 100-epoch model successfully generated a latent error heatmap. However, it revealed a classic ViT **Low-Frequency Bias**. Because the simulated anomaly was dark and uniform, the model predicted it easily. It instead flagged the bright, complex speckle of the healthy myocardium as the "error." This proved the infrastructure works perfectly, while highlighting that ViTs require significantly more scale (500+ epochs) and Variance Regularization to master high-frequency textures.

---

##  Future Roadmap
Building on this JAX baseline, the next stages of development include:

1. **Variance-Covariance Regularization:** Modifying the latent loss function to mathematically force the embeddings to capture high-fidelity tissue textures, eliminating the low-frequency bias.
2. **Spatiotemporal Video-JEPA:** Upgrading the 2D Encoders to 3D ViTs, allowing the model to ingest sequential frames and predict imminent arrhythmias based on temporal biomechanical rhythms.
3. **Action-Conditioned Surgical Sandbox:** Injecting action vectors (representing robotic actuator impedance or acoustic transducer focus) alongside the context tokens to predict soft-tissue deformation *before* a physical intervention occurs.

## License
MIT License.
