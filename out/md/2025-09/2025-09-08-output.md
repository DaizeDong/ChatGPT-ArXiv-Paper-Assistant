# Personalized Daily ArXiv Papers 2025-09-08

| *[gpt-5]*   | Prompt   | Completion   | Total   |
|:-----------:|:--------:|:------------:|:-------:|
| **Token**   | 39153    | 40799        | 79952   |
| **Cost**    | $0.05    | $0.41        | $0.46   |

Total arXiv papers: 381

Total scanned papers: 243

Total relevant papers: 20

**Table of contents with paper titles:**

1. [SpikingBrain Technical Report: Spiking Brain-inspired Large Models](#user-content-link1)
**Authors:** Yuqi Pan, Yupeng Feng, Jinghao Zhuang, Siyu Ding, Zehao Liu, Bohan Sun, Yuhong Chou, Han Xu, Xuerui Qiu, Anlin Deng, Anjie Hu, Peng Zhou, Man Yao, Jibin Wu, Jian Yang, Guoliang Sun, Bo Xu, Guoqi Li

2. [KVCompose: Efficient Structured KV Cache Compression with Composite Tokens](#user-content-link2)
**Authors:** Dmitry Akulov, Mohamed Sana, Antonio De Domenico, Tareq Si Salem, Nicola Piovesan, Fadhel Ayed

3. [HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models](#user-content-link3)
**Authors:** Chang Dai, Hongyu Shan, Mingyang Song, Di Liang

4. [Interpreting Transformer Architectures as Implicit Multinomial Regression](#user-content-link4)
**Authors:** Jonas A. Actor, Anthony Gruber, Eric C. Cyr

5. [Just-in-time and distributed task representations in language models](#user-content-link5)
**Authors:** Yuxuan Li, Declan Campbell, Stephanie C. Y. Chan, Andrew Kyle Lampinen

6. [Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference](#user-content-link6)
**Authors:** Hao Zhang, Mengsi Lyu, Yulong Ao, Yonghua Lin

7. [Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](#user-content-link7)
**Authors:** Deniz Bayazit, Aaron Mueller, Antoine Bosselut

8. [Dynamical Learning in Deep Asymmetric Recurrent Neural Networks](#user-content-link8)
**Authors:** Davide Badalotti, Carlo Baldassi, Marc M\'ezard, Mattia Scardecchia, Riccardo Zecchina

9. [Sample-efficient Integration of New Modalities into Large Language Models](#user-content-link9)
**Authors:** Osman Batur \.Ince, Andr\'e F. T. Martins, Oisin Mac Aodha, Edoardo M. Ponti

10. [Beyond I-Con: Exploring New Dimension of Distance Measures in Representation Learning](#user-content-link10)
**Authors:** Jasmine Shone, Shaden Alshammari, Mark Hamilton, Zhening Li, William Freeman

11. [Probabilistic operator learning: generative modeling and uncertainty quantification for foundation models of differential equations](#user-content-link11)
**Authors:** Benjamin J. Zhang, Siting Liu, Stanley J. Osher, Markos A. Katsoulakis

12. [Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions](#user-content-link12)
**Authors:** Faruk Alpay, Taylan Alpay

13. [VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation](#user-content-link13)
**Authors:** Mustafa Munir, Alex Zhang, Radu Marculescu

14. [Natural Spectral Fusion: p-Exponent Cyclic Scheduling and Early Decision-Boundary Alignment in First-Order Optimization](#user-content-link14)
**Authors:** Gongyue Zhang, Honghai Liu

15. [HyPINO: Multi-Physics Neural Operators via HyperPINNs and the Method of Manufactured Solutions](#user-content-link15)
**Authors:** Rafael Bischof, Michal Piovar\v{c}i, Michael A. Kraus, Siddhartha Mishra, Bernd Bickel

16. [Any-Step Density Ratio Estimation via Interval-Annealed Secant Alignment](#user-content-link16)
**Authors:** Wei Chen, Shigui Li, Jiacheng Li, Jian Xu, Zhiqi Lin, Junmei Yang, Delu Zeng, John Paisley, Qibin Zhao

17. [ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute](#user-content-link17)
**Authors:** Hao Wen, Yifan Su, Feifei Zhang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, Yuanchun Li

18. [Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](#user-content-link18)
**Authors:** Fangzhou Wu, Sandeep Silwal

19. [Neuro-Spectral Architectures for Causal Physics-Informed Networks](#user-content-link19)
**Authors:** Arthur Bizzi, Leonardo M. Moreira, M\'arcio Marques, Leonardo Mendon\c{c}a, Christian J\'unior de Oliveira, Vitor Balestro, Lucas dos Santos Fernandez, Daniel Yukimura, Pavel Petrov, Jo\~ao M. Pereira, Tiago Novello, Lucas Nissenbaum

20. [Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization](#user-content-link20)
**Authors:** Shuaicheng Niu, Guohao Chen, Deyu Chen, Yifan Zhang, Jiaxiang Wu, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Chunyan Miao, Mingkui Tan

---

## 1. [SpikingBrain Technical Report: Spiking Brain-inspired Large Models](https://arxiv.org/abs/2509.05276) <a id="link1"></a>

**ArXiv ID:** 2509.05276

**Authors:** Yuqi Pan, Yupeng Feng, Jinghao Zhuang, Siyu Ding, Zehao Liu, Bohan Sun, Yuhong Chou, Han Xu, Xuerui Qiu, Anlin Deng, Anjie Hu, Peng Zhou, Man Yao, Jibin Wu, Jian Yang, Guoliang Sun, Bo Xu, Guoqi Li

**Abstract:** Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware.   Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design.

**Comment:** Model Architecture, Compression/Efficiency, and HPC: spiking LLMs with linear/hybrid-linear attention and MoE, sparse/event-driven inference with near-constant memory, and custom distributed training on non-NVIDIA hardware.

**Relevance:** 10
**Novelty:** 9

---

## 2. [KVCompose: Efficient Structured KV Cache Compression with Composite Tokens](https://arxiv.org/abs/2509.05165) <a id="link2"></a>

**ArXiv ID:** 2509.05165

**Authors:** Dmitry Akulov, Mohamed Sana, Antonio De Domenico, Tareq Si Salem, Nicola Piovesan, Fadhel Ayed

**Abstract:** Large language models (LLMs) rely on key-value (KV) caches for efficient autoregressive decoding; however, cache size grows linearly with context length and model depth, becoming a major bottleneck in long-context inference. Prior KV cache compression methods either enforce rigid heuristics, disrupt tensor layouts with per-attention-head variability, or require specialized compute kernels.   We propose a simple, yet effective, KV cache compression framework based on attention-guided, layer-adaptive composite tokens. Our method aggregates attention scores to estimate token importance, selects head-specific tokens independently, and aligns them into composite tokens that respect the uniform cache structure required by existing inference engines. A global allocation mechanism further adapts retention budgets across layers, assigning more capacity to layers with informative tokens. This approach achieves significant memory reduction while preserving accuracy, consistently outperforming prior structured and semi-structured methods. Crucially, our approach remains fully compatible with standard inference pipelines, offering a practical and scalable solution for efficient long-context LLM deployment.

**Comment:** Model Compression and Efficiency: structured KV cache compression via attention-guided, layer-adaptive composite tokens compatible with standard inference engines.

**Relevance:** 10
**Novelty:** 8

---

## 3. [HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models](https://arxiv.org/abs/2509.05218) <a id="link3"></a>

**ArXiv ID:** 2509.05218

**Authors:** Chang Dai, Hongyu Shan, Mingyang Song, Di Liang

**Abstract:** Positional encoding mechanisms enable Transformers to model sequential structure and long-range dependencies in text. While absolute positional encodings struggle with extrapolation to longer sequences due to fixed positional representations, and relative approaches like Alibi exhibit performance degradation on extremely long contexts, the widely-used Rotary Positional Encoding (RoPE) introduces oscillatory attention patterns that hinder stable long-distance dependency modelling. We address these limitations through a geometric reformulation of positional encoding. Drawing inspiration from Lorentz transformations in hyperbolic geometry, we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations. Theoretical analysis demonstrates that RoPE is a special case of our generalized formulation. HoPE fundamentally resolves RoPE's slation issues by enforcing monotonic decay of attention weights with increasing token distances. Extensive experimental results, including perplexity evaluations under several extended sequence benchmarks, show that HoPE consistently exceeds existing positional encoding methods. These findings underscore HoPE's enhanced capacity for representing and generalizing long-range dependencies. Data and code will be available.

**Comment:** Model Architecture: introduces hyperbolic rotary positional encoding (HoPE), a geometric generalization of RoPE for stable long-range dependencies.

**Relevance:** 10
**Novelty:** 8

---

## 4. [Interpreting Transformer Architectures as Implicit Multinomial Regression](https://arxiv.org/abs/2509.04653) <a id="link4"></a>

**ArXiv ID:** 2509.04653

**Authors:** Jonas A. Actor, Anthony Gruber, Eric C. Cyr

**Abstract:** Mechanistic interpretability aims to understand how internal components of modern machine learning models, such as weights, activations, and layers, give rise to the model's overall behavior. One particularly opaque mechanism is attention: despite its central role in transformer models, its mathematical underpinnings and relationship to concepts like feature polysemanticity, superposition, and model performance remain poorly understood. This paper establishes a novel connection between attention mechanisms and multinomial regression. Specifically, we show that in a fixed multinomial regression setting, optimizing over latent features yields optimal solutions that align with the dynamics induced by attention blocks. In other words, the evolution of representations through a transformer can be interpreted as a trajectory that recovers the optimal features for classification.

**Comment:** Model Architecture/Mechanistic Interpretability: establishes a theoretical link between attention dynamics in transformers and optimal feature recovery in multinomial regression.

**Relevance:** 10
**Novelty:** 8

---

## 5. [Just-in-time and distributed task representations in language models](https://arxiv.org/abs/2509.04466) <a id="link5"></a>

**ArXiv ID:** 2509.04466

**Authors:** Yuxuan Li, Declan Campbell, Stephanie C. Y. Chan, Andrew Kyle Lampinen

**Abstract:** Many of language models' impressive capabilities originate from their in-context learning: based on instructions or examples, they can infer and perform new tasks without weight updates. In this work, we investigate \emph{when} representations for new tasks are formed in language models, and \emph{how} these representations change over the course of context. We focus on ''transferrable'' task representations -- vector representations that can restore task context in another instance of the model, even without the full prompt. We show that these representations evolve in non-monotonic and sporadic ways, and are distinct from a more inert representation of high-level task categories that persists throughout the context. Specifically, models often condense multiple evidence into these transferrable task representations, which align well with the performance improvement based on more examples in the context. However, this accrual process exhibits strong locality along the sequence dimension, coming online only at certain tokens -- despite task identity being reliably decodable throughout the context. Moreover, these local but transferrable task representations tend to capture minimal ''task scopes'', such as a semantically-independent subtask, and models rely on more temporally-distributed representations to support longer and composite tasks. This two-fold locality (temporal and semantic) underscores a kind of just-in-time computational process underlying language models' ability to adapt to new evidence and learn new tasks on the fly.

**Comment:** Matches Representation Learning criterion—empirical analysis of when/where transferable task representations form and evolve during in-context learning in LMs.

**Relevance:** 9
**Novelty:** 8

---

## 6. [Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference](https://arxiv.org/abs/2509.04467) <a id="link6"></a>

**ArXiv ID:** 2509.04467

**Authors:** Hao Zhang, Mengsi Lyu, Yulong Ao, Yonghua Lin

**Abstract:** Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the default settings, our method achieves a 20.56% inference speedup and a 4.95 times reduction in data transmission bandwidth consumption.

**Comment:** Strongly matches Model Compression and Efficiency—stage-specific block pruning for prefill vs. decode and token-aware KV cache pruning tailored to PD disaggregation.

**Relevance:** 9
**Novelty:** 8

---

## 7. [Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining](https://arxiv.org/abs/2509.05291) <a id="link7"></a>

**ArXiv ID:** 2509.05291

**Authors:** Deniz Bayazit, Aaron Mueller, Antoine Bosselut

**Abstract:** Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining.

**Comment:** Representation Learning: aligns sparse features across checkpoints and introduces a causal metric (RelIE) to track emergence/maintenance of linguistic features during pretraining.

**Relevance:** 9
**Novelty:** 8

---

## 8. [Dynamical Learning in Deep Asymmetric Recurrent Neural Networks](https://arxiv.org/abs/2509.05041) <a id="link8"></a>

**ArXiv ID:** 2509.05041

**Authors:** Davide Badalotti, Carlo Baldassi, Marc M\'ezard, Mattia Scardecchia, Riccardo Zecchina

**Abstract:** We show that asymmetric deep recurrent neural networks, enhanced with additional sparse excitatory couplings, give rise to an exponentially large, dense accessible manifold of internal representations which can be found by different algorithms, including simple iterative dynamics. Building on the geometrical properties of the stable configurations, we propose a distributed learning scheme in which input-output associations emerge naturally from the recurrent dynamics, without any need of gradient evaluation. A critical feature enabling the learning process is the stability of the configurations reached at convergence, even after removal of the supervisory output signal. Extensive simulations demonstrate that this approach performs competitively on standard AI benchmarks. The model can be generalized in multiple directions, both computational and biological, potentially contributing to narrowing the gap between AI and computational neuroscience.

**Comment:** Matches Model Architecture (asymmetric deep recurrent networks with sparse excitatory couplings) and Representation Learning (emergent stable manifolds and gradient-free distributed learning dynamics).

**Relevance:** 9
**Novelty:** 8

---

## 9. [Sample-efficient Integration of New Modalities into Large Language Models](https://arxiv.org/abs/2509.04606) <a id="link9"></a>

**ArXiv ID:** 2509.04606

**Authors:** Osman Batur \.Ince, Andr\'e F. T. Martins, Oisin Mac Aodha, Edoardo M. Ponti

**Abstract:** Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector -- placed between modality-specific encoders and an LLM -- to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially multiply the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models.

**Comment:** Model architecture: a hypernetwork generates adapters for a shared modality-agnostic projector, enabling few-shot integration of new modalities into an LLM (sample-efficient modality extension of foundation models).

**Relevance:** 9
**Novelty:** 8

---

## 10. [Beyond I-Con: Exploring New Dimension of Distance Measures in Representation Learning](https://arxiv.org/abs/2509.04734) <a id="link10"></a>

**ArXiv ID:** 2509.04734

**Authors:** Jasmine Shone, Shaden Alshammari, Mark Hamilton, Zhening Li, William Freeman

**Abstract:** The Information Contrastive (I-Con) framework revealed that over 23 representation learning methods implicitly minimize KL divergence between data and learned distributions that encode similarities between data points. However, a KL-based loss may be misaligned with the true objective, and properties of KL divergence such as asymmetry and unboundedness may create optimization challenges. We present Beyond I-Con, a framework that enables systematic discovery of novel loss functions by exploring alternative statistical divergences and similarity kernels. Key findings: (1) on unsupervised clustering of DINO-ViT embeddings, we achieve state-of-the-art results by modifying the PMI algorithm to use total variation (TV) distance; (2) on supervised contrastive learning, we outperform the standard approach by using TV and a distance-based similarity kernel instead of KL and an angular kernel; (3) on dimensionality reduction, we achieve superior qualitative results and better performance on downstream tasks than SNE by replacing KL with a bounded f-divergence. Our results highlight the importance of considering divergence and similarity kernel choices in representation learning optimization.

**Comment:** Matches Representation Learning: introduces a framework exploring alternative divergences and similarity kernels to derive new objectives beyond KL (contrastive/PMI/SNE).

**Relevance:** 9
**Novelty:** 8

---

## 11. [Probabilistic operator learning: generative modeling and uncertainty quantification for foundation models of differential equations](https://arxiv.org/abs/2509.05186) <a id="link11"></a>

**ArXiv ID:** 2509.05186

**Authors:** Benjamin J. Zhang, Siting Liu, Stanley J. Osher, Markos A. Katsoulakis

**Abstract:** In-context operator networks (ICON) are a class of operator learning methods based on the novel architectures of foundation models. Trained on a diverse set of datasets of initial and boundary conditions paired with corresponding solutions to ordinary and partial differential equations (ODEs and PDEs), ICON learns to map example condition-solution pairs of a given differential equation to an approximation of its solution operator. Here, we present a probabilistic framework that reveals ICON as implicitly performing Bayesian inference, where it computes the mean of the posterior predictive distribution over solution operators conditioned on the provided context, i.e., example condition-solution pairs. The formalism of random differential equations provides the probabilistic framework for describing the tasks ICON accomplishes while also providing a basis for understanding other multi-operator learning methods. This probabilistic perspective provides a basis for extending ICON to \emph{generative} settings, where one can sample from the posterior predictive distribution of solution operators. The generative formulation of ICON (GenICON) captures the underlying uncertainty in the solution operator, which enables principled uncertainty quantification in the solution predictions in operator learning.

**Comment:** Representation Learning: probabilistic/Bayesian framework for operator foundation models (ICON) with generative posterior and uncertainty quantification.

**Relevance:** 8
**Novelty:** 8

---

## 12. [Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions](https://arxiv.org/abs/2509.04549) <a id="link12"></a>

**ArXiv ID:** 2509.04549

**Authors:** Faruk Alpay, Taylan Alpay

**Abstract:** Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models.

**Comment:** Model Architecture/Representation Learning: principled activation- and weight-space interventions with theory for controllable transformers.

**Relevance:** 8
**Novelty:** 7

---

## 13. [VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation](https://arxiv.org/abs/2509.04669) <a id="link13"></a>

**ArXiv ID:** 2509.04669

**Authors:** Mustafa Munir, Alex Zhang, Radu Marculescu

**Abstract:** Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce \textit{VCMamba}, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3% with 37% fewer parameters, and outperforming Vision GNN-B by 0.3% with 64% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62% fewer parameters. Code is available at https://github.com/Wertyuui345/VCMamba.

**Comment:** Matches Model Architecture criterion—hybrid CNN + multi-directional Mamba SSM backbone with hierarchical design for linear-complexity global modeling.

**Relevance:** 8
**Novelty:** 7

---

## 14. [Natural Spectral Fusion: p-Exponent Cyclic Scheduling and Early Decision-Boundary Alignment in First-Order Optimization](https://arxiv.org/abs/2509.04713) <a id="link14"></a>

**ArXiv ID:** 2509.04713

**Authors:** Gongyue Zhang, Honghai Liu

**Abstract:** Spectral behaviors have been widely discussed in machine learning, yet the optimizer's own spectral bias remains unclear. We argue that first-order optimizers exhibit an intrinsic frequency preference that significantly reshapes the optimization path. To address this, we propose Natural Spectral Fusion (NSF): reframing training as controllable spectral coverage and information fusion rather than merely scaling step sizes. NSF has two core principles: treating the optimizer as a spectral controller that dynamically balances low- and high-frequency information; and periodically reweighting frequency bands at negligible cost, without modifying the model, data, or training pipeline. We realize NSF via a p-exponent extension of the second-moment term, enabling both positive and negative exponents, and implement it through cyclic scheduling. Theory and experiments show that adaptive methods emphasize low frequencies, SGD is near-neutral, and negative exponents amplify high-frequency information. Cyclic scheduling broadens spectral coverage, improves cross-band fusion, and induces early decision-boundary alignment, where accuracy improves even while loss remains high. Across multiple benchmarks, with identical learning-rate strategies and fixed hyperparameters, p-exponent cyclic scheduling consistently reduces test error and demonstrates distinct convergence behavior; on some tasks, it matches baseline accuracy with only one-quarter of the training cost. Overall, NSF reveals the optimizer's role as an active spectral controller and provides a unified, controllable, and efficient framework for first-order optimization.

**Comment:** Matches Representation Learning (training dynamics via spectral bias analysis) and Model Efficiency (cyclic p-exponent adaptive-moment scheduling that reduces training cost).

**Relevance:** 8
**Novelty:** 7

---

## 15. [HyPINO: Multi-Physics Neural Operators via HyperPINNs and the Method of Manufactured Solutions](https://arxiv.org/abs/2509.05117) <a id="link15"></a>

**ArXiv ID:** 2509.05117

**Authors:** Rafael Bischof, Michal Piovar\v{c}i, Michael A. Kraus, Siddhartha Mishra, Bernd Bickel

**Abstract:** We present HyPINO, a multi-physics neural operator designed for zero-shot generalization across a broad class of parametric PDEs without requiring task-specific fine-tuning. Our approach combines a Swin Transformer-based hypernetwork with mixed supervision: (i) labeled data from analytical solutions generated via the Method of Manufactured Solutions (MMS), and (ii) unlabeled samples optimized using physics-informed objectives. The model maps PDE parametrizations to target Physics-Informed Neural Networks (PINNs) and can handle linear elliptic, hyperbolic, and parabolic equations in two dimensions with varying source terms, geometries, and mixed Dirichlet/Neumann boundary conditions, including interior boundaries. HyPINO achieves strong zero-shot accuracy on seven benchmark problems from PINN literature, outperforming U-Nets, Poseidon, and Physics-Informed Neural Operators (PINO). Further, we introduce an iterative refinement procedure that compares the physics of the generated PINN to the requested PDE and uses the discrepancy to generate a "delta" PINN. Summing their contributions and repeating this process forms an ensemble whose combined solution progressively reduces the error on six benchmarks and achieves over 100x gain in average $L_2$ loss in the best case, while retaining forward-only inference. Additionally, we evaluate the fine-tuning behavior of PINNs initialized by HyPINO and show that they converge faster and to lower final error than both randomly initialized and Reptile-meta-learned PINNs on five benchmarks, performing on par on the remaining two. Our results highlight the potential of this scalable approach as a foundation for extending neural operators toward solving increasingly complex, nonlinear, and high-dimensional PDE problems with significantly improved accuracy and reduced computational cost.

**Comment:** Architecture: hypernetwork-based neural operator that maps PDE specs to PINNs with mixed supervision and iterative refinement; advances zero-shot operator generalization with computational gains.

**Relevance:** 7
**Novelty:** 8

---

## 16. [Any-Step Density Ratio Estimation via Interval-Annealed Secant Alignment](https://arxiv.org/abs/2509.04852) <a id="link16"></a>

**ArXiv ID:** 2509.04852

**Authors:** Wei Chen, Shigui Li, Jiacheng Li, Jian Xu, Zhiqi Lin, Junmei Yang, Delu Zeng, John Paisley, Qibin Zhao

**Abstract:** Estimating density ratios is a fundamental problem in machine learning, but existing methods often trade off accuracy for efficiency. We propose \textit{Interval-annealed Secant Alignment Density Ratio Estimation (ISA-DRE)}, a framework that enables accurate, any-step estimation without numerical integration.   Instead of modeling infinitesimal tangents as in prior methods, ISA-DRE learns a global secant function, defined as the expectation of all tangents over an interval, with provably lower variance, making it more suitable for neural approximation. This is made possible by the \emph{Secant Alignment Identity}, a self-consistency condition that formally connects the secant with its underlying tangent representations.   To mitigate instability during early training, we introduce \emph{Contraction Interval Annealing}, a curriculum strategy that gradually expands the alignment interval during training. This process induces a contraction mapping, which improves convergence and training stability.   Empirically, ISA-DRE achieves competitive accuracy with significantly fewer function evaluations compared to prior methods, resulting in much faster inference and making it well suited for real-time and interactive applications.

**Comment:** Algorithmic efficiency: any-step density ratio estimation via a Secant Alignment identity and interval annealing, eliminating numerical integration and reducing function evaluations.

**Relevance:** 7
**Novelty:** 8

---

## 17. [ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute](https://arxiv.org/abs/2509.04475) <a id="link17"></a>

**ArXiv ID:** 2509.04475

**Authors:** Hao Wen, Yifan Su, Feifei Zhang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, Yuanchun Li

**Abstract:** Recent advances in Large Language Models (LLMs) have been driven by test-time compute scaling - a strategy that improves reasoning by generating longer, sequential thought processes. While effective, this approach encounters a significant bottleneck as computation increases, where further computation offers only marginal performance gains. We argue this ceiling is not an inherent limit of the model's capability but a flaw in the scaling strategy itself, a phenomenon we term "Tunnel Vision", where a model's imperfect initial steps lock it into a suboptimal reasoning path. To overcome this, we introduce a new scaling paradigm: native thought parallelism. We present ParaThinker, an end-to-end framework that trains an LLM to generate multiple, diverse reasoning paths in parallel and synthesize them into a superior final answer. By exploring different lines of thoughts simultaneously, ParaThinker effectively sidesteps the Tunnel Vision issue and unlocks the model's latent reasoning potential. Our approach demonstrates that scaling compute in parallel (width) is a more effective and efficient way to superior reasoning than simply scaling sequentially (depth). On challenging reasoning benchmarks, ParaThinker achieves substantial accuracy improvements over sequential LLMs (12.3% for 1.5B and 7.5% for 7B models on average with 8 parallel paths), while adding only negligible latency overhead (7.1%). This enables smaller models to surpass much larger counterparts and establishes parallel thinking as a critical, efficient dimension for scaling future LLMs.

**Comment:** Model architecture/efficiency at inference: trains LLMs for native parallel reasoning paths to scale test-time compute in width.

**Relevance:** 7
**Novelty:** 7

---

## 18. [Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](https://arxiv.org/abs/2509.02718) <a id="link18"></a>

**ArXiv ID:** 2509.02718

**Authors:** Fangzhou Wu, Sandeep Silwal

**Abstract:** Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput.

**Comment:** Matches High Performance Computing/systems criterion—training-free online LLM routing with ANN-based feature estimation and competitive-ratio guarantees for high-throughput serving.

**Relevance:** 7
**Novelty:** 7

---

## 19. [Neuro-Spectral Architectures for Causal Physics-Informed Networks](https://arxiv.org/abs/2509.04966) <a id="link19"></a>

**ArXiv ID:** 2509.04966

**Authors:** Arthur Bizzi, Leonardo M. Moreira, M\'arcio Marques, Leonardo Mendon\c{c}a, Christian J\'unior de Oliveira, Vitor Balestro, Lucas dos Santos Fernandez, Daniel Yukimura, Pavel Petrov, Jo\~ao M. Pereira, Tiago Novello, Lucas Nissenbaum

**Abstract:** Physics-Informed Neural Networks (PINNs) have emerged as a powerful neural framework for solving partial differential equations (PDEs). However, standard MLP-based PINNs often fail to converge when dealing with complex initial-value problems, leading to solutions that violate causality and suffer from a spectral bias towards low-frequency components. To address these issues, we introduce NeuSA (Neuro-Spectral Architectures), a novel class of PINNs inspired by classical spectral methods, designed to solve linear and nonlinear PDEs with variable coefficients. NeuSA learns a projection of the underlying PDE onto a spectral basis, leading to a finite-dimensional representation of the dynamics which is then integrated with an adapted Neural ODE (NODE). This allows us to overcome spectral bias, by leveraging the high-frequency components enabled by the spectral representation; to enforce causality, by inheriting the causal structure of NODEs, and to start training near the target solution, by means of an initialization scheme based on classical methods. We validate NeuSA on canonical benchmarks for linear and nonlinear wave equations, demonstrating strong performance as compared to other architectures, with faster convergence, improved temporal consistency and superior predictive accuracy. Code and pretrained models will be released.

**Comment:** Matches Model Architecture (neuro-spectral PINN combining spectral projection with Neural ODE) and Representation Learning (addresses spectral bias and causal consistency).

**Relevance:** 7
**Novelty:** 7

---

## 20. [Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization](https://arxiv.org/abs/2509.04977) <a id="link20"></a>

**ArXiv ID:** 2509.04977

**Authors:** Shuaicheng Niu, Guohao Chen, Deyu Chen, Yifan Zhang, Jiaxiang Wu, Zhiquan Wen, Yaofo Chen, Peilin Zhao, Chunyan Miao, Mingkui Tan

**Abstract:** Test-time adaptation (TTA) may fail to improve or even harm the model performance when test data have: 1) mixed distribution shifts, 2) small batch sizes, 3) online imbalanced label distribution shifts. This is often a key obstacle preventing existing TTA methods from being deployed in the real world. In this paper, we investigate the unstable reasons and find that the batch norm layer is a crucial factor hindering TTA stability. Conversely, TTA can perform more stably with batch-agnostic norm layers, i.e., group or layer norm. However, we observe that TTA with group and layer norms does not always succeed and still suffers many failure cases, i.e., the model collapses into trivial solutions by assigning the same class label for all samples. By digging into this, we find that, during the collapse process: 1) the model gradients often undergo an initial explosion followed by rapid degradation, suggesting that certain noisy test samples with large gradients may disrupt adaptation; and 2) the model representations tend to exhibit high correlations and classification bias. To address this, we first propose a sharpness-aware and reliable entropy minimization method, called SAR, for stabilizing TTA from two aspects: 1) remove partial noisy samples with large gradients, 2) encourage model weights to go to a flat minimum so that the model is robust to the remaining noisy samples. Based on SAR, we further introduce SAR^2 to prevent representation collapse with two regularizers: 1) a redundancy regularizer to reduce inter-dimensional correlations among centroid-invariant features; and 2) an inequity regularizer to maximize the prediction entropy of a prototype centroid, thereby penalizing biased representations toward any specific class. Promising results demonstrate that our methods perform more stably over prior methods and are computationally efficient under the above wild test scenarios.

**Comment:** Matches Model Efficiency (stable test-time adaptation via sharpness-aware updates) and Representation Learning (feature regularizers to prevent representation collapse).

**Relevance:** 7
**Novelty:** 7

---

# Paper Selection Prompt

## System Prompt

> You are a helpful paper reading assistant whose job is to read daily posts from ArXiv and identify a few papers that your friend will enjoy reading.
> Your job is to carefully read the paper titles and abstracts below and find the ones that match the criteria below.

## User Prompt

> ## Instructions
> 
> Write the response in JSONL format with {ARXIVID, COMMENT, RELEVANCE, NOVELTY} on each line, one for each paper.
> 
> - ARXIVID: should be the ArXiv ID.
> - COMMENT: should identify whether there is a criteria that match the paper very closely. These matches should not be based on general terms like "language modeling" or "advancements" and should specifically refer to a criterion. No need to mention the non-matching criteria.
> - RELEVANCE: should be a score from 1-10.
> - NOVELTY: should be a score from 1-10.
> 
> ## Scoring Criteria
> 
> > The "Relevance" score measures how closely the paper aligns with the core topics of the prompt.
> > The "Novelty" score assesses the originality and impact of the paper.
> > They are two **ORTHONORMAL** axes and **SHOULD NOT** be confused with each other.
> 
> ### Relevance Scoring
> 
> - Relevance 9-10 (Completely Relevant)
>   - Focus: Fully aligned with core topics with no deviation, score the highest if contains relevant keywords in it.
>   - Examples: Papers focused on foundational methods or theoretical research, whose titles contain topic keywords like "MoE".
> 
> - Relevance 7-8 (Relevant)
>   - Focus: Retain a solid link to the main research area, though may touch on peripheral elements.
>   - Examples: Papers research on the fundamental part of MoE through a less critical aspect like its behavior in GNN.
> 
> - Relevance 5-6 (Borderline)
>   - Focus: Maintains a link to the core topic but also extends into at least one other domain/area beyond the primary focus.
>   - Examples: Work referencing MoE centered on reinforcement learning.
> 
> - Relevance 3-4 (Irrelevant)
>   - Focus: Largely outside our interests with no association to our topics.
>   - Examples: Application-focused papers like using MoE to solve a problem in the real world.
> 
> - Relevance 1-2 (Ignore)
>   - Focus: Purely unrelated to our topics. Completely a different domain.
>   - **Exception**: If the paper hints at a cutting-edge, radically new direction that could eventually transform the primary domain, consider a score of 9–10 despite initial appearances. (Usually a very rare concept that belongs to the fundamental research)
> 
> ### Novelty Scoring
> 
> - Novelty 9-10 (Breakthrough)
>   - Definition: Groundbreaking methods/theory introducing new directions or solving major challenges.
>   - Examples: Entirely new paradigm for foundational models; a novel theory transforming representation learning.
> 
> - Novelty 7-8 (Improvements)
>   - Definition: Substantial insights/enhancements, though not a full paradigm shift.
>   - Examples: Modifications on existing methods yielding significantly better results.
> 
> - Novelty 5-6 (Borderline)
>   - Definition: Incremental contributions with possible long-term benefits, not immediately transformative.
>   - Examples: Moderately novel extension to an existing architecture; refining current methods without fundamentally altering them.
> 
> - Novelty 3-4 (Tangential)
>   - Definition: Minor or domain-specific improvements with limited broader impact.
>   - Examples: Slight modifications to known methods with strange motivation; purely engineering jobs like a new benchmark/dataset.
> 
> - Novelty 1-2 (Low)
>   - Definition: Minimal originality, applying standard approaches without real innovation.
>   - Examples: Using an off-the-shelf model without adding new insights; purely application-driven studies like finetuning a pretrained model using existing methods.
> 
> ## Papers
> 
> [PAPER LIST HERE]
> 
> ## Relevant Topics
> 
> Use the following relevance criteria to focus on foundational research. Keep **relevant** papers and filter out **irrelevant** ones. Avoid purely **application-driven** work.
> 
> 1. Model Architecture
>    - Relevant: Mixture-of-Experts (MoE), Transformers, Conditional/Dynamic Networks, Autoencoders, analysis/innovations on existing architectures.
>    - Irrelevant: Merely using existing architectures for a certain task without insights into the structure themselves.
> 
> 2. Model Compression and Efficiency
>    - Relevant: Sparsity, pruning, quantization, low-rank approaches, cache, or other algorithmic/theoretical efficiency breakthroughs.
>    - Irrelevant: Straightforward applications of existing compression methods to new tasks.
> 
> 3. High Performance Computing
>    - Relevant: Algorithmic or systems-level innovations enabling training of large-scale models, distributed training techniques, memory optimization.
>    - Irrelevant: Incremental engineering improvements without novel algorithmic contributions.
> 
> 4. Representation Learning
>    - Relevant: Insights into how deep networks encode information, feature/dictionary learning, sparse/contrastive methods, training dynamics in neural networks.
>    - Irrelevant: Standard applications of known techniques lacking new theoretical or methodological contributions.
> 
> **Keywords:**
> 
> - Relevant: Mixture of Experts (MoE), Representation Learning, Compression/Efficiency, Sparse/Sparsity, Pruning, Quantization, Low-rank, Foundation Model, etc.
> - Irrelevant: Reinforcement Learning, Transfer Learning, Federated Learning, Online Learning, Diffusion Models, etc.
> - Application: Image Segmentation, Medical Imaging, 3D Vision, Video Understanding, Information Retrieval, Summarization, Recommendation Systems, Machine Translation, Speech Recognition, Signal Processing, Spatial/Temporal Modeling, Time Series, Knowledge Graph, etc.