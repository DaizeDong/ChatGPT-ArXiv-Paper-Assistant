<div align="center"><a href="../../2025-02/summary">&larr; Previous Summary</a>&nbsp;|&nbsp;<a href="..">Monthly Overview</a>&nbsp;|&nbsp;<a href="../../2025-04/summary">Next Summary &rarr;</a><br><a href="../../2025-02/summary">2025-02</a>&nbsp;|&nbsp;<a href="..">2025-03</a>&nbsp;|&nbsp;<a href="../../2025-04/summary">2025-04</a></div>

# Personalized Monthly Topic Summary 2025/03

<table>
    <thead>
        <tr><th>Metric</th><th>Value</th></tr>
    </thead>
    <tbody>
        <tr><td><strong>Total Papers</strong></td><td align="center">18</td></tr>
        <tr><td>Frontier Model Releases and Technical Reports</td><td align="center">1</td></tr>
        <tr><td>Architecture and Training Dynamics</td><td align="center">12</td></tr>
        <tr><td>Training Algorithms That Change What Is Possible</td><td align="center">2</td></tr>
        <tr><td>MoE Where It Changes the Design Space</td><td align="center">2</td></tr>
        <tr><td>Efficiency, Compression, and Large-Scale Training</td><td align="center">1</td></tr>
        <tr><td>Representation Learning Theory and Structure</td><td align="center">0</td></tr>
        <tr><td>Memory Structures and Agent Memory Systems</td><td align="center">0</td></tr>
        <tr><td>World Models, Exploration, and Open-Ended Reinforcement Learning</td><td align="center">0</td></tr>
    </tbody>
</table>

## Frontier Model Releases and Technical Reports (1)

1. [RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://arxiv.org/abs/2503.14456)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-03-19](../19)
   - **Comment:** Generalized delta updates with vector-valued gates expand recurrent state expressivity while preserving parallelizable training.

## Architecture and Training Dynamics (12)

1. [Transformers without Normalization](https://arxiv.org/abs/2503.10622)
   - **Score:** 20 (R=10, N=10)
   - **Date:** [2025-03-14](../14)
   - **Comment:** Replaces normalization with an element-wise learned tanh operation while reporting preserved or improved Transformer performance.

2. [SuperBPE: Space Travel for Language Models](https://arxiv.org/abs/2503.13423)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-03-18](../18)
   - **Comment:** Removes word-boundary tokenization constraints while holding model size, vocabulary size, and pretraining compute fixed.

3. [Predictable Scale: Part I -- Optimal Hyperparameter Scaling Law in Large Language Model Pretraining](https://arxiv.org/abs/2503.04715)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-03-07](../07)
   - **Comment:** Maps optimal pretraining learning rate and batch size against parameter count and training-token budget.

4. [Forgetting Transformer: Softmax Attention with a Forget Gate](https://arxiv.org/abs/2503.02130)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-03-05](../05)
   - **Comment:** Data-dependent attention forgetting removes the need for positional embeddings.

5. [Computation Mechanism Behind LLM Position Generalization](https://arxiv.org/abs/2503.13305)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-18](../18)
   - **Comment:** Finds a learned additive separation of positional relevance and semantic importance in attention logits.

6. [Outlier dimensions favor frequent tokens in language model](https://arxiv.org/abs/2503.21718)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-03-28](../28)
   - **Comment:** Identifies final-layer outliers as a frequent-token prediction mechanism and explains the counterweights that suppress it.

7. [Generalized Interpolating Discrete Diffusion](https://arxiv.org/abs/2503.04482)
   - **Score:** 18 (R=10, N=8)
   - **Date:** [2025-03-07](../07)
   - **Comment:** Generalizes discrete-diffusion training beyond absorbing masks, allowing generated tokens to be revised.

8. [I Predict Therefore I Am: Is Next Token Prediction Enough to Learn Human-Interpretable Concepts from Data?](https://arxiv.org/abs/2503.08980)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-13](../13)
   - **Comment:** Derives when next-token training produces linear representations of latent-concept posteriors.

9. [Interpreting the Repeated Token Phenomenon in Large Language Models](https://arxiv.org/abs/2503.08908)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-13](../13)
   - **Comment:** Identifies the attention-sink circuit disrupted by repetition and tests a targeted repair.

10. [Strategy Coopetition Explains the Emergence and Transience of In-Context Learning](https://arxiv.org/abs/2503.05631)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-10](../10)
   - **Comment:** Shared subcircuits make context-constrained in-weights learning both enable and eventually displace in-context learning.

11. [(How) Do Language Models Track State?](https://arxiv.org/abs/2503.02854)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-05](../05)
   - **Comment:** Training interventions select between associative-scan and parity-assisted state-tracking algorithms.

12. [Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries](https://arxiv.org/abs/2502.20475)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-03](../03)
   - **Comment:** Identifies a promote-then-suppress computation shared by attention and MLPs, tested through token-specific knockouts.

## Training Algorithms That Change What Is Possible (2)

1. [Compute Optimal Scaling of Skills: Knowledge vs Reasoning](https://arxiv.org/abs/2503.10061)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-03-14](../14)
   - **Comment:** Finds skill-dependent compute-optimal scaling after controlling for pretraining data mixture, changing how model size should be selected.

2. [Training LLMs with MXFP4](https://arxiv.org/abs/2502.20586)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-03-03](../03)
   - **Comment:** Combines stochastic rounding with Hadamard transforms to control MXFP4 gradient variance during pretraining up to 6.7B parameters.

## MoE Where It Changes the Design Space (2)

1. [Continual Pre-training of MoEs: How robust is your router?](https://arxiv.org/abs/2503.05029)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-10](../10)
   - **Comment:** Four large MoEs retain router balance through continual pretraining, including without replay.

2. [Mixture of Lookup Experts](https://arxiv.org/abs/2503.15798)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-21](../21)
   - **Comment:** Token-embedding-only experts become lookup tables, eliminating their inference-time FFN computation.

## Efficiency, Compression, and Large-Scale Training (1)

1. [Language Models May Verbatim Complete TextThey Were Not Explicitly Trained On](https://arxiv.org/abs/2503.17514)
   - **Score:** 18 (R=9, N=9)
   - **Date:** [2025-03-25](../25)
   - **Comment:** Removal-and-retraining counterexamples show that verbatim completion does not establish n-gram-defined training membership.