<div align="center"><a href="../../2025-01/summary">&larr; Previous Summary</a>&nbsp;|&nbsp;<a href="..">Monthly Overview</a>&nbsp;|&nbsp;<a href="../../2025-03/summary">Next Summary &rarr;</a><br><a href="../../2025-01/summary">2025-01</a>&nbsp;|&nbsp;<a href="..">2025-02</a>&nbsp;|&nbsp;<a href="../../2025-03/summary">2025-03</a></div>

# Personalized Monthly Topic Summary 2025/02

<table>
    <thead>
        <tr><th>Metric</th><th>Value</th></tr>
    </thead>
    <tbody>
        <tr><td><strong>Total Papers</strong></td><td align="center">19</td></tr>
        <tr><td>Frontier Model Releases and Technical Reports</td><td align="center">0</td></tr>
        <tr><td>Architecture and Training Dynamics</td><td align="center">12</td></tr>
        <tr><td>Training Algorithms That Change What Is Possible</td><td align="center">3</td></tr>
        <tr><td>MoE Where It Changes the Design Space</td><td align="center">3</td></tr>
        <tr><td>Efficiency, Compression, and Large-Scale Training</td><td align="center">1</td></tr>
        <tr><td>Representation Learning Theory and Structure</td><td align="center">0</td></tr>
        <tr><td>Memory Structures and Agent Memory Systems</td><td align="center">0</td></tr>
        <tr><td>World Models, Exploration, and Open-Ended Reinforcement Learning</td><td align="center">0</td></tr>
    </tbody>
</table>

## Architecture and Training Dynamics (12)

1. [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-02-25](../25)
   - **Comment:** Holds effective depth at kL while replacing kL distinct layers with k layers reused L times.

2. [The underlying structures of self-attention: symmetry, directionality, and emergent dynamics in Transformer training](https://arxiv.org/abs/2502.10927)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-02-18](../18)
   - **Comment:** Derives how training objectives produce symmetric versus directional structure in self-attention weights.

3. [Which Attention Heads Matter for In-Context Learning?](https://arxiv.org/abs/2502.14010)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-02-21](../21)
   - **Comment:** Ablations across 12 language models distinguish function-vector and induction-head contributions and track their relationship during training.

4. [Systematic Outliers in Large Language Models](https://arxiv.org/abs/2502.06415)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-02-11](../11)
   - **Comment:** Explains systematic outliers as implicit context-aware scaling induced by attention softmax.

5. [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-02-10](../10)
   - **Comment:** Decouples stored parameter count from computation depth by repeatedly applying a pretrained recurrent block.

6. [MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections](https://arxiv.org/abs/2502.12170)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-02-19](../19)
   - **Comment:** Gives query, key, value, and residual streams separate input-dependent mixtures of earlier layers.

7. [Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations](https://arxiv.org/abs/2502.18147)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-02-26](../26)
   - **Comment:** Tests whether computational sparsity is learned by comparing pretrained transformers with randomized counterparts.

8. [Neural Attention: A Novel Mechanism for Enhanced Expressive Power in Transformer Models](https://arxiv.org/abs/2502.17206)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-02-25](../25)
   - **Comment:** Replaces dot-product attention scoring with learned feed-forward networks at unchanged attention-matrix dimensions.

9. [Continuous Diffusion Model for Language Modeling](https://arxiv.org/abs/2502.11564)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-02-18](../18)
   - **Comment:** Connects categorical diffusion to continuous manifold flow and derives simulation-free training.

10. [Prediction hubs are context-informed frequent tokens in LLMs](https://arxiv.org/abs/2502.10201)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-02-17](../17)
   - **Comment:** Distinguishes frequency-driven prediction hubs from distance-concentration artifacts at the unembedding readout.

11. [Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers](https://arxiv.org/abs/2502.00213)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-02-04](../04)
   - **Comment:** Connects Adam's advantage to gradient-norm heterogeneity and its dependence on layer-normalization placement.

12. [Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing](https://arxiv.org/abs/2502.19416)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-02-27](../27)
   - **Comment:** Localized updates increase weight norms while shrinking and shifting activations, exposing instability in layer balance.

## Training Algorithms That Change What Is Possible (3)

1. [LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws](https://arxiv.org/abs/2502.12120)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-02-18](../18)
   - **Comment:** Finds that data and tokenizer determine loss-to-loss curves while architecture and optimizer choices largely leave them unchanged.

2. [SkipPipe: Partial and Reordered Pipelining Framework for Training LLMs in Heterogeneous Networks](https://arxiv.org/abs/2502.19913)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-02-28](../28)
   - **Comment:** Derives convergence constraints that allow training microbatches to skip and reorder pipeline stages.

3. [QuEST: Stable Training of LLMs with 1-Bit Weights and Activations](https://arxiv.org/abs/2502.05003)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-02-10](../10)
   - **Comment:** Introduces a trust gradient estimator that stabilizes LLM training with 1-bit weights and activations.

## MoE Where It Changes the Design Space (3)

1. [Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient](https://arxiv.org/abs/2502.05172)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-02-10](../10)
   - **Comment:** Models active parameters, expert count and data jointly to configure MoE training under fixed memory and compute budgets.

2. [Scaling Laws for Upcycling Mixture-of-Experts Language Models](https://arxiv.org/abs/2502.03009)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-02-06](../06)
   - **Comment:** Models the interaction between dense-pretraining and upcycled-MoE token budgets to identify when reuse beats training from scratch.

3. [Mixture of Tunable Experts - Behavior Modification of DeepSeek-R1 at Inference Time](https://arxiv.org/abs/2502.11096)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-02-18](../18)
   - **Comment:** Random deactivation and forced activation test whether specific experts causally control localized behavior.

## Efficiency, Compression, and Large-Scale Training (1)

1. [Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models](https://arxiv.org/abs/2501.19090)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-02-03](../03)
   - **Comment:** Losslessly removes redundancy from low-rank representations, reducing storage while holding the represented layer's function fixed.