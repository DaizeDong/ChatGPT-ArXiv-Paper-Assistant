# Personalized Daily Arxiv Papers 01/21/2025
Total cost: $1.026625

Total relevant papers: 10

Paper selection prompt and criteria at the bottom

Table of contents with paper titles:

0. [Accelerating Large Language Models through Partially Linear Feed-Forward Network](#user-content-link0)
**Authors:** Gansen Hu, Zhaoguo Wang, Jinglin Wei, Wei Huang, Haibo Chen

1. [LeMo: Enabling LEss Token Involvement for MOre Context Fine-tuning](#user-content-link1)
**Authors:** Tuowei Wang, Xingyu Chen, Kun Li, Ting Cao, Ju Ren, Yaoxue Zhang

2. [MultiPruner: Balanced Structure Removal in Foundation Models](#user-content-link2)
**Authors:** J. Pablo Mu\~noz, Jinjie Yuan, Nilesh Jain

3. [Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models](#user-content-link3)
**Authors:** Qiang Liu, Xinlong Chen, Yue Ding, Shizhen Xu, Shu Wu, Liang Wang

4. [AIRCHITECT v2: Learning the Hardware Accelerator Design Space through Unified Representations](#user-content-link4)
**Authors:** Jamin Seo, Akshat Ramachandran, Yu-Chuan Chuang, Anirudh Itagi, Tushar Krishna

5. [OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking](#user-content-link5)
**Authors:** Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang, Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, Huajun Chen

6. [Evolving Deeper LLM Thinking](#user-content-link6)
**Authors:** Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, Xinyun Chen

7. [Enhancing Generalization in Chain of Thought Reasoning for Smaller Models](#user-content-link7)
**Authors:** Maxwell J. Yin, Dingyi Jiang, Yongbing Chen, Boyu Wang, Charles Ling

8. [Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning](#user-content-link8)
**Authors:** Yirong Zeng, Xiao Ding, Yuxian Wang, Weiwen Liu, Wu Ning, Yutai Hou, Xu Huang, Bing Qin, Ting Liu

9. [Good things come in small packages: Should we adopt Lite-GPUs in AI infrastructure?](#user-content-link9)
**Authors:** Burcu Canakci, Junyi Liu, Xingbo Wu, Nathana\"el Cheriere, Paolo Costa, Sergey Legtchenko, Dushyanth Narayanan, Ant Rowstron

---
## 0. [Accelerating Large Language Models through Partially Linear Feed-Forward Network](https://arxiv.org/abs/2501.10054) <a id="link0"></a>

**ArXiv ID:** 2501.10054

**Authors:** Gansen Hu, Zhaoguo Wang, Jinglin Wei, Wei Huang, Haibo Chen

**Abstract:** Large language models (LLMs) demonstrate remarkable capabilities but face deployment challenges due to their massive parameter counts. While existing compression techniques like pruning can reduce model size, it leads to significant accuracy degradation under high compression ratios. We present a novel perspective inspired by constant folding in compiler optimization. Our approach enables parameter reduction by treating activation functions in LLMs as linear functions.   However, recent LLMs use complex non-linear activations like GELU that prevent direct application of this technique. We propose TARDIS, which enables optimization of LLMs with non-linear activations by partially approximating them with linear functions in frequently occurring input ranges. For outlier inputs, TARDIS employs an online predictor to dynamically fall back to original computations.   Our experiments demonstrate that TARDIS achieves 80% parameter reduction in feed-forward networks, while significantly outperforming state-of-the-art pruning methods Wanda and RIA with up to 65% higher accuracy. In practical deployments for a 7B model, TARDIS achieves 1.6x end-to-end inference speedup when integrated with the vLLM serving system, and 1.4x speedup with the widely adopted HuggingFace implementation, while incurring only a 10.9% accuracy trade-off.

**Comment:** The paper proposes TARDIS, a novel method for compressing feed-forward networks in LLMs by leveraging partial linear approximations, which ties closely to the model compression topic with innovative insights into efficiency improvements.

**Relevance:** 9
**Novelty:** 8

---

## 1. [LeMo: Enabling LEss Token Involvement for MOre Context Fine-tuning](https://arxiv.org/abs/2501.09767) <a id="link1"></a>

**ArXiv ID:** 2501.09767

**Authors:** Tuowei Wang, Xingyu Chen, Kun Li, Ting Cao, Ju Ren, Yaoxue Zhang

**Abstract:** The escalating demand for long-context applications has intensified the necessity of extending the LLM context windows. Despite recent fine-tuning approaches successfully expanding context lengths, their high memory footprints, especially for activations, present a critical practical limitation. Current parameter-efficient fine-tuning methods prioritize reducing parameter update overhead over addressing activation memory constraints. Similarly, existing sparsity mechanisms improve computational efficiency but overlook activation memory optimization due to the phenomenon of Shadowy Activation.   In this paper, we propose LeMo, the first LLM fine-tuning system that explores and exploits a new token-level sparsity mechanism inherent in long-context scenarios, termed Contextual Token Sparsity. LeMo minimizes redundant token involvement by assessing the informativeness of token embeddings while preserving model accuracy. Specifically, LeMo introduces three key techniques: (1) Token Elimination, dynamically identifying and excluding redundant tokens across varying inputs and layers. (2) Pattern Prediction, utilizing well-trained predictors to approximate token sparsity patterns with minimal overhead. (3) Kernel Optimization, employing permutation-free and segment-based strategies to boost system performance. We implement LeMo as an end-to-end fine-tuning system compatible with various LLM architectures and other optimization techniques. Comprehensive evaluations demonstrate that LeMo reduces memory consumption by up to 1.93x and achieves up to 1.36x speedups, outperforming state-of-the-art fine-tuning systems.

**Comment:** Proposes a fine-tuning system for LLMs addressing activation memory constraints using token-level sparsity. Relevant to the compression and efficiency domain of LLMs, and includes novel memory-related optimization techniques.

**Relevance:** 9
**Novelty:** 8

---

## 2. [MultiPruner: Balanced Structure Removal in Foundation Models](https://arxiv.org/abs/2501.09949) <a id="link2"></a>

**ArXiv ID:** 2501.09949

**Authors:** J. Pablo Mu\~noz, Jinjie Yuan, Nilesh Jain

**Abstract:** Recently, state-of-the-art approaches for pruning large pre-trained models (LPMs) have demonstrated that the training-free removal of non-critical residual blocks in Transformers is viable for reducing model size, achieving results that outperform previous training-free pruning approaches. Motivated by these findings, we extend BlockPruner (Zhong et al., 2024) and propose MultiPruner, a pruning approach that surpasses recent training-free pruning methods by adopting a multidimensional, iterative, fine-grained pruning strategy. In MultiPruner, multidimensional pruning reinstates the structural balance in block-pruned models by sequentially compressing along three dimensions: i) residual blocks, ii) channels of multilayer perceptrons (MLP), and iii) attention heads. This solution enhances zero-shot accuracy on downstream tasks compared to other techniques while improving model compression ratios, producing compressed models with fewer computing and memory requirements. Extensive experiments demonstrate the advantages of the proposed method across various large pre-trained models. The code and pruning configurations are available at https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning.

**Comment:** This paper introduces MultiPruner, which enhances model compression strategies by adopting a multi-dimensional, balanced pruning approach. It directly targets model compression with structural and algorithmic innovation, aligning well with the core topics.

**Relevance:** 9
**Novelty:** 8

---

## 3. [Attention-guided Self-reflection for Zero-shot Hallucination Detection in Large Language Models](https://arxiv.org/abs/2501.09997) <a id="link3"></a>

**ArXiv ID:** 2501.09997

**Authors:** Qiang Liu, Xinlong Chen, Yue Ding, Shizhen Xu, Shu Wu, Liang Wang

**Abstract:** Hallucination has emerged as a significant barrier to the effective application of Large Language Models (LLMs). In this work, we introduce a novel Attention-Guided SElf-Reflection (AGSER) approach for zero-shot hallucination detection in LLMs. The AGSER method utilizes attention contributions to categorize the input query into attentive and non-attentive queries. Each query is then processed separately through the LLMs, allowing us to compute consistency scores between the generated responses and the original answer. The difference between the two consistency scores serves as a hallucination estimator. In addition to its efficacy in detecting hallucinations, AGSER notably reduces computational complexity, requiring only three passes through the LLM and utilizing two sets of tokens. We have conducted extensive experiments with four widely-used LLMs across three different hallucination benchmarks, demonstrating that our approach significantly outperforms existing methods in zero-shot hallucination detection.

**Comment:** The paper proposes a novel attention-guided self-reflection (AGSER) method for zero-shot hallucination detection in LLMs. It aligns with foundational insights into LLM behavior and efficiency, fitting well into topics like sparsity and innovative architectural features for error mitigation.

**Relevance:** 9
**Novelty:** 8

---

## 4. [AIRCHITECT v2: Learning the Hardware Accelerator Design Space through Unified Representations](https://arxiv.org/abs/2501.09954) <a id="link4"></a>

**ArXiv ID:** 2501.09954

**Authors:** Jamin Seo, Akshat Ramachandran, Yu-Chuan Chuang, Anirudh Itagi, Tushar Krishna

**Abstract:** Design space exploration (DSE) plays a crucial role in enabling custom hardware architectures, particularly for emerging applications like AI, where optimized and specialized designs are essential. With the growing complexity of deep neural networks (DNNs) and the introduction of advanced foundational models (FMs), the design space for DNN accelerators is expanding at an exponential rate. Additionally, this space is highly non-uniform and non-convex, making it increasingly difficult to navigate and optimize. Traditional DSE techniques rely on search-based methods, which involve iterative sampling of the design space to find the optimal solution. However, this process is both time-consuming and often fails to converge to the global optima for such design spaces. Recently, AIrchitect v1, the first attempt to address the limitations of search-based techniques, transformed DSE into a constant-time classification problem using recommendation networks. In this work, we propose AIrchitect v2, a more accurate and generalizable learning-based DSE technique applicable to large-scale design spaces that overcomes the shortcomings of earlier approaches. Specifically, we devise an encoder-decoder transformer model that (a) encodes the complex design space into a uniform intermediate representation using contrastive learning and (b) leverages a novel unified representation blending the advantages of classification and regression to effectively explore the large DSE space without sacrificing accuracy. Experimental results evaluated on 10^5 real DNN workloads demonstrate that, on average, AIrchitect v2 outperforms existing techniques by 15% in identifying optimal design points. Furthermore, to demonstrate the generalizability of our method, we evaluate performance on unseen model workloads (LLMs) and attain a 1.7x improvement in inference latency on the identified hardware architecture.

**Comment:** AIrchitect v2 proposes a transformer-based approach for learning hardware design spaces, addressing scalability and efficiency in DNN accelerator optimization. This is relevant to model efficiency and emerging trends in foundational AI for hardware applications.

**Relevance:** 8
**Novelty:** 8

---

## 5. [OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking](https://arxiv.org/abs/2501.09751) <a id="link5"></a>

**ArXiv ID:** 2501.09751

**Authors:** Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang, Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, Huajun Chen

**Abstract:** Machine writing with large language models often relies on retrieval-augmented generation. However, these approaches remain confined within the boundaries of the model's predefined scope, limiting the generation of content with rich information. Specifically, vanilla-retrieved information tends to lack depth, utility, and suffers from redundancy, which negatively impacts the quality of generated articles, leading to shallow, repetitive, and unoriginal outputs. To address these issues, we propose OmniThink, a machine writing framework that emulates the human-like process of iterative expansion and reflection. The core idea behind OmniThink is to simulate the cognitive behavior of learners as they progressively deepen their knowledge of the topics. Experimental results demonstrate that OmniThink improves the knowledge density of generated articles without compromising metrics such as coherence and depth. Human evaluations and expert feedback further highlight the potential of OmniThink to address real-world challenges in the generation of long-form articles.

**Comment:** OmniThink proposes a framework for iterative knowledge expansion in LLMs, emulating human-like cognitive processes for long-form content generation. This aligns with LLM theoretical topics and introduces novel insights into enhancing knowledge density in outputs.

**Relevance:** 8
**Novelty:** 8

---

## 6. [Evolving Deeper LLM Thinking](https://arxiv.org/abs/2501.09891) <a id="link6"></a>

**ArXiv ID:** 2501.09891

**Authors:** Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, Xinyun Chen

**Abstract:** We explore an evolutionary search strategy for scaling inference time compute in Large Language Models. The proposed approach, Mind Evolution, uses a language model to generate, recombine and refine candidate responses. The proposed approach avoids the need to formalize the underlying inference problem whenever a solution evaluator is available. Controlling for inference cost, we find that Mind Evolution significantly outperforms other inference strategies such as Best-of-N and Sequential Revision in natural language planning tasks. In the TravelPlanner and Natural Plan benchmarks, Mind Evolution solves more than 98% of the problem instances using Gemini 1.5 Pro without the use of a formal solver.

**Comment:** The paper introduces 'Mind Evolution' for scaling inference time compute in LLMs. The evolutionary search strategy and problem-solving insights show considerable relevance to scaling and inference cost strategies in LLMs, though foundational breakthroughs are limited.

**Relevance:** 8
**Novelty:** 7

---

## 7. [Enhancing Generalization in Chain of Thought Reasoning for Smaller Models](https://arxiv.org/abs/2501.09804) <a id="link7"></a>

**ArXiv ID:** 2501.09804

**Authors:** Maxwell J. Yin, Dingyi Jiang, Yongbing Chen, Boyu Wang, Charles Ling

**Abstract:** Chain-of-Thought (CoT) reasoning in smaller language models is a challenging natural language process problem yet highly desirable in many real-life applications. Existing CoT knowledge distillation methods often suffer from overly conservative memorization in smaller LLMs, leading to low generalization confidence. As fully preserving the CoT ability of teacher model is impossible, we hypothesize that adversarial CoT fine-tuning is crucial for developing smaller LLM with robust CoT generalization. To this end, we propose \textit{PRompt-Assisted Domain-Adversarial fine-tuning} (PRADA), a principled fine-tuning framework that integrates diverse CoT domains. Specifically, PRADA pioneers two CoT improvements in smaller LLM: (1) Recovering the domain-invariant feature insight which typically lost during distillation with domain adversarial fine-tuning; (2) Enhancing the domain adaptability of CoT prompt engineering by employing domain-adversarial approaches. We theoretically demonstrate the effectiveness of our approach and empirically show that it significantly outperforms the state of the arts in a wide range of tasks. Moreover, our empirical findings reveal that the smaller LLM, when leveraging PRADA, aligns closely with domain knowledge, thereby improving the explainability of our approach.

**Comment:** Proposes PRADA, which focuses on enhancing chain-of-thought reasoning in smaller LLMs via adversarial finetuning. This has relevance in representation learning and LLM efficiency, though it is not a paradigm shift and mainly extends existing techniques.

**Relevance:** 7
**Novelty:** 7

---

## 8. [Boosting Tool Use of Large Language Models via Iterative Reinforced Fine-Tuning](https://arxiv.org/abs/2501.09766) <a id="link8"></a>

**ArXiv ID:** 2501.09766

**Authors:** Yirong Zeng, Xiao Ding, Yuxian Wang, Weiwen Liu, Wu Ning, Yutai Hou, Xu Huang, Bing Qin, Ting Liu

**Abstract:** Augmenting large language models (LLMs) with external tools is a promising approach to enhance their capabilities. Effectively leveraging this potential for complex tasks hinges crucially on improving their ability to use tools. Synthesizing tool use data by simulating the real world is an effective approach. Nevertheless, our investigation reveals that training gains significantly decay as the scale of these data increases. The primary factor is the model's poor performance (a.k.a deficiency) in complex scenarios, which hinders learning from data using SFT. Driven by this objective, we propose an iterative reinforced fine-tuning strategy to continually guide the model to alleviate it. Specifically, we first identify deficiency-related data based on feedback from the policy model, then perform a Monte Carlo Tree Search to collect fine-grained preference pairs to pinpoint deficiencies. Subsequently, we update the policy model using preference optimization to align with ground truth and misalign with deficiencies. This process can be iterated. Moreover, before the iteration, we propose an easy-to-hard warm-up SFT strategy to facilitate learning from challenging data. The experiments demonstrate our models go beyond the same parametric models, outperforming many larger open-source and closed-source models. Additionally, it has achieved notable training gains in complex tool use scenarios.

**Comment:** The paper discusses iterative reinforced fine-tuning to address deficiencies in complex tool-use scenarios for LLMs. It aligns somewhat with the Large Language Models (LLMs) topic, focusing on training advancements like iterative fine-tuning but lacks foundational or architectural breakthroughs.

**Relevance:** 7
**Novelty:** 7

---

## 9. [Good things come in small packages: Should we adopt Lite-GPUs in AI infrastructure?](https://arxiv.org/abs/2501.10187) <a id="link9"></a>

**ArXiv ID:** 2501.10187

**Authors:** Burcu Canakci, Junyi Liu, Xingbo Wu, Nathana\"el Cheriere, Paolo Costa, Sergey Legtchenko, Dushyanth Narayanan, Ant Rowstron

**Abstract:** To match the blooming demand of generative AI workloads, GPU designers have so far been trying to pack more and more compute and memory into single complex and expensive packages. However, there is growing uncertainty about the scalability of individual GPUs and thus AI clusters, as state-of-the-art GPUs are already displaying packaging, yield, and cooling limitations. We propose to rethink the design and scaling of AI clusters through efficiently-connected large clusters of Lite-GPUs, GPUs with single, small dies and a fraction of the capabilities of larger GPUs. We think recent advances in co-packaged optics can be key in overcoming the communication challenges of distributing AI workloads onto more Lite-GPUs. In this paper, we present the key benefits of Lite-GPUs on manufacturing cost, blast radius, yield, and power efficiency; and discuss systems opportunities and challenges around resource, workload, memory, and network management.

**Comment:** The paper proposes the use of Lite-GPUs to address scalability and efficiency in AI clusters. This potentially links to model compression and scaling themes, which are relevant topics. However, the focus is more on hardware-level innovations rather than core algorithmic or architectural insights.

**Relevance:** 7
**Novelty:** 7

---


---

# Paper selection prompt
You are a helpful paper reading assistant whose job is to read daily posts from ArXiv and identify a few papers that your friend will enjoy reading.
Your job is to carefully read the paper titles and abstracts below and find the ones that match the criteria below.

## Relevant Topics

1. Representation Learning
   - Relevant: Feature learning, sparse/contrastive learning, dictionary learning, or theoretical insights into how deep networks encode information.
   - Irrelevant: Application-only work using standard representation learning without innovative insights.

2. Model Architecture
   - Relevant: Mixture-of-Experts (MoE), Transformers, Conditional/Dynamic Networks, Autoencoders, and other foundational structures.
   - Irrelevant: Simply applying existing architectures to new tasks without structural/theoretical innovation.

3. Model Compression
   - Relevant: Sparsity, pruning, quantization, low-rank, KV cache, or theoretical/algorithmic innovations for efficiency, etc.
   - Irrelevant: Simply applying existing compression to new tasks.

4. Large Language Models (LLMs)
   - Relevant: Strong theoretical insights on LLM behavior, architecture/training breakthroughs (e.g., MoE).
   - Irrelevant: Domain-specific usage or small tweaks (e.g., instruction tuning), lack of theoretical advancement (e.g., benchmarks/datasets, inference tricks like RAG).

5. AI for Science
   - Relevant: Foundational research in molecule/protein modeling (e.g., new training paradigms, advanced generative methods, or theoretical perspectives), or major architecture-level innovation.
   - Irrelevant: Conventional, domain-limited applications lacking insights on the foundational side.

6. Emerging Trends
   - Relevant: Cutting-edge theoretical work challenging assumptions, or broad new paradigms/concepts in AI research.
   - Irrelevant: Trend-following or incremental extensions on existing methods.

**Note: Foundation vs. Application**
   - Foundational/theoretical papers (new theorems, architectures, or strong methodological insights) are of **high** relevance.
   - Subdomain papers and application-focused papers (e.g., "methods for xxx") are **lower** in relevance.

**Hints on Irrelevant Domains:**
Federated Learning, Online Learning, Transfer Learning, Reinforcement Learning, etc.

**Hints on Application Tasks:**
Image Segmentation, Medical Imaging, Speech Recognition, Video Understanding, Recommendation Systems, 3D Vision, Machine Translation, Information Retrieval, etc.


## Papers

[PAPER LIST HERE]

## Scoring Criteria

> The "Relevance" score measures how closely the paper aligns with the core topics of the prompt.
> The "Novelty" score assesses the originality and impact of the paper.
> They are two **ORTHONORMAL** axes and **SHOULD NOT** be confused with each other. E.g., a paper with high relevance can be of low novelty, or vice versa.

### Relevance Scoring

- Relevance 9-10 (Completely Relevant)
  - Focus: Fully aligned with core topics, score the highest if also contains keywords in it.
  - Keywords: “Mixture of Experts (MoE),” “Representation Learning,” “Compression,” “Sparse/Sparsity,” “Pruning,” “Quantization,” “Low-rank,” “Scaling,” “Foundation Models,” etc.
  - Examples: Papers focused on foundational methods or theoretical research, whose titles contain topic keywords like "MoE".

- Relevance 7-8 (Relevant)
  - Focus: Clearly tied to our main topics, may not fully hit the interest in foundational methods.
  - Examples: Pure research on representation/architecture on MoE with no other domain focus.

- Relevance 5-6 (Optional)
  - Focus: Link to our topics—covers relevant ideas but also includes another area of interest.
  - Examples: Work referencing MoE centered on another domain.

- Relevance 3-4 (Irrelevant)
  - Focus: Largely outside our interests, with no association to our topics.
  - Examples: Application-focused papers like using MoE to solve a problem in real world.

- Relevance 1-2 (Ignore)
  - Focus: Purely unrelated to our topics. Completely a different domain.
  - Exception: If you think it is an emerging trend (that may lead to a thorough breakthrough in the future), you can give a score of 9-10. (Usually a very rare concept that belongs to the fundamental research)

### Novelty Scoring

- Novelty 9-10 (Breakthrough)
  - Definition: Groundbreaking methods/theory introducing new directions or solving major challenges.
  - Examples: Entirely new paradigm for foundational models; a novel theory transforming representation learning.

- Novelty 7-8 (Improvements)
  - Definition: Substantial insights/enhancements, though not a full paradigm shift.
  - Examples: Modifications on existing methods yielding significantly better results.

- Novelty 5-6 (Moderate)
  - Definition: Incremental contributions with possible long-term benefits, not immediately transformative.
  - Examples: Moderately novel extension to an existing architecture; refining current methods without fundamentally altering them.

- Novelty 3-4 (Tangential)
  - Definition: Minor or domain-specific improvements with limited broader impact.
  - Examples: Slight modifications to known methods with strange motivation; purely engineering jobs like a new benchmark/dataset.

- Novelty 1-2 (Low)
  - Definition: Minimal originality, applying standard approaches without real innovation.
  - Examples: Using an off-the-shelf model without adding new insights; purely application-driven studies like finetuning a pretrained model using existing methods.


## Instructions

Write the response in JSONL format with {ARXIVID, COMMENT, RELEVANCE, NOVELTY} on each line, one for each paper.

- ARXIVID: should be the ArXiv ID.
- COMMENT: should identify whether there is a criteria that match the paper very closely. These matches should not be based on general terms like "language modeling" or "advancements" and should specifically refer to a criterion. No need to mention the non-matching criteria.
- RELEVANCE: should be a score from 1-10.
- NOVELTY: should be a score from 1-10.