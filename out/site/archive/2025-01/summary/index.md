<div align="center"><a href="..">Monthly Overview</a>&nbsp;|&nbsp;<a href="../../2025-02/summary">Next Summary &rarr;</a><br><a href="..">2025-01</a>&nbsp;|&nbsp;<a href="../../2025-02/summary">2025-02</a></div>

# Personalized Monthly Topic Summary 2025/01

<table>
    <thead>
        <tr><th>Metric</th><th>Value</th></tr>
    </thead>
    <tbody>
        <tr><td><strong>Total Papers</strong></td><td align="center">4</td></tr>
        <tr><td>Frontier Model Releases and Technical Reports</td><td align="center">0</td></tr>
        <tr><td>Architecture and Training Dynamics</td><td align="center">2</td></tr>
        <tr><td>Training Algorithms That Change What Is Possible</td><td align="center">0</td></tr>
        <tr><td>MoE Where It Changes the Design Space</td><td align="center">2</td></tr>
        <tr><td>Efficiency, Compression, and Large-Scale Training</td><td align="center">0</td></tr>
        <tr><td>Representation Learning Theory and Structure</td><td align="center">0</td></tr>
        <tr><td>Memory Structures and Agent Memory Systems</td><td align="center">0</td></tr>
        <tr><td>World Models, Exploration, and Open-Ended Reinforcement Learning</td><td align="center">0</td></tr>
    </tbody>
</table>

## Architecture and Training Dynamics (2)

1. [Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling](https://arxiv.org/abs/2501.16975)
   - **Score:** 20 (R=10, N=10)
   - **Date:** [2025-01-29](../29)
   - **Comment:** Decouples input and output vocabularies, scaling multi-gram input capacity while keeping the output vocabulary fixed.

2. [Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing for Robust, Adaptable Language Models](https://arxiv.org/abs/2501.10322)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-01-20](../20)
   - **Comment:** Replaces the fixed subword vocabulary with character-to-word encoding and character decoding, demonstrated at up to 7B parameters.

## MoE Where It Changes the Design Space (2)

1. [Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models](https://arxiv.org/abs/2501.12370)
   - **Score:** 17 (R=9, N=8)
   - **Date:** [2025-01-22](../22)
   - **Comment:** Varies MoE sparsity under parameter or training-compute constraints to separate stored capacity from active compute.

2. [Autonomy-of-Experts Models](https://arxiv.org/abs/2501.13074)
   - **Score:** 19 (R=10, N=9)
   - **Date:** [2025-01-23](../23)
   - **Comment:** Removes the standalone MoE router and selects experts using their own internal activation norms.