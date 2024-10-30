<div align="center">
<img src="./figure/logo.jpg" style="width: 20%;height: 10%">
<h1> LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models </h1>
</div>

<div align="center">

</div>

<div align="center">
  <!-- <a href="#model">Model</a> • -->
  🏆 <a href="https://collam.megatechai.com/">Leaderboard</a> |
  📚 <a href="https://huggingface.co/datasets/CSHaitao/LexEval">Data</a> |
  📃 <a href="https://arxiv.org/abs/2409.20288">Paper</a> |
  📊 <a href="#citation">Citation</a>
</div>


Welcome to **LexEval**, the comprehensive Chinese legal benchmark designed to evaluate the performance of large language models (LLMs) in legal applications.

*Read this in [中文] (README_ZH.md).*


## What's New
- **[2024.10]** 🥳 LexEval is accepted by NeurIPs 2024

## Introduction

Large language models (LLMs) have made significant progress in natural language processing tasks and have shown considerable potential in the legal domain.  However, the legal applications often have high requirements on accuracy, reliability and fairness. Applying existing LLMs to legal systems without careful evaluation of their potentials and limitations could lead to significant risks in legal practice.
Therefore, to facilitate the healthy development and application of LLMs in the legal domain, we propose a comprehensive benchmark LexEval for evaluating LLMs in legal domain. 

Key aspects of LexEval:

**Ability Modeling:** We propose a novel Legal Cognitive Ability Taxonomy (LexAbility Taxonomy) to organize different legal tasks systematically. This taxonomy includes six core abilities: Memorization, Understanding, Logic Inference, Discrimination, Generation, and Ethics.

**Scale:** LexEval is currently the largest legal benchmark in China, comprising 23 tasks and 14,150 questions. Additionally, LexEval will be continuously updated to enable more comprehensive evaluations.

**Data Sources:** LexEval combines data from existing legal datasets, real-world exam datasets, and newly annotated datasets created by legal experts to provide a comprehensive assessment of LLM capabilities.

## Legal Cognitive Ability Taxonomy (LexCog)

Inspired by Bloom's taxonomy and real-world legal application scenarios, we propose a legal cognitive ability taxonomy (LexCog) to provide guidance for the evaluation of LLMs. Our taxonomy categorizes the application of LLMs in the legal domain into six ability levels: Memorization, Understanding, Logic Inference, Discrimination, Generation, and Ethic. 

<div align="center">

<img src="./figure/taxonomy.png">
<!-- <h1> A nice pic from our website </h1> -->

</div>

- **Memorization**: At this level, LLMs are responsible for recalling and accurately retrieving key legal information, such as fundamental statutes, case laws, legal principles, and specialized terminology. This ability ensures that models can store and access foundational knowledge relevant to legal tasks.

- **Understanding**: LLMs are tested on their ability to comprehend and interpret legal texts and concepts. They must grasp the meaning, implications, and relevance of legal content, demonstrating an ability to accurately understand the issues presented in legal cases, documents, or regulations.

- **Logic Inference**: This ability evaluates the model’s capacity for legal reasoning and deduction. LLMs are expected to apply logical reasoning to derive conclusions from given legal facts and rules. This includes identifying patterns, drawing inferences, and making connections between legal principles in a structured manner.

- **Discrimination**: LLMs must exhibit the ability to analyze, compare, and evaluate the significance of legal information based on specific legal criteria. This involves distinguishing between similar legal concepts, analyzing case precedents, and determining the relevance of legal evidence.

- **Generation**: At this level, LLMs are expected to produce professional and legally sound documents. This includes drafting legal opinions, contracts, case summaries, and other legal texts. The generated content should be precise, well-structured, and adhere to legal standards based on provided instructions or conditions.

- **Ethics**: The Ethics level assesses the model's capacity to recognize and address ethical issues in legal contexts. LLMs should be able to analyze legal ethical dilemmas, weigh pros and cons, and make decisions aligned with professional ethics, legal principles, and social values.

This taxonomy serves as the foundation of LexEval, ensuring that LLMs are evaluated comprehensively across the full spectrum of legal cognitive tasks. By organizing tasks into these six levels, the LexEval benchmark provides a robust framework for assessing how well LLMs can support legal professionals in their work, while also highlighting potential areas for improvement in the models' abilities to handle complex legal scenarios.


## Tasks Definition

The dataset for Lexeval consists of 14,150 questions carefully designed to cover the breadth of legal cognitive abilities outlined in the LexCog. The questions span 23 tasks relevant to legal scenarios, providing a diverse set for evaluating LLM performance.

The following table shows the details of the tasks in LexEval:
![image](./figure/tasks.png)

Further experimental details and analyses can be found in our paper.


## Contributing

LexEval is an ongoing project, and we welcome contributions from the community. You can contribute by:

* Adding new tasks to the LexAbility Taxonomy

* Submitting new datasets or annotations

* Improving the evaluation framework

Please contact liht22@mails.tsinghua.edu.cn

## License

LexEval is released under the [MIT License](LICENSE).


## Citation
If you find our work useful, please do not save your star and cite our work:

```bibtex
@misc{li2024lexevalcomprehensivechineselegal,
      title={LexEval: A Comprehensive Chinese Legal Benchmark for Evaluating Large Language Models}, 
      author={Haitao Li and You Chen and Qingyao Ai and Yueyue Wu and Ruizhe Zhang and Yiqun Liu},
      year={2024},
      eprint={2409.20288},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.20288}, 
}
```
