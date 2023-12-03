
<div align="center">
  <img src="docs/imgs/llama-falcon.png"  width="500">
</div>

# Megatron-LLM

This library enables pre-training and fine-tuning of large language models (LLMs) at scale.
Our repository is a modification of the [original Megatron-LM codebase](https://github.com/NVIDIA/Megatron-LM) by Nvidia.

Added key features include:
- architectures supported: [Llama](https://arxiv.org/abs/2302.13971), [Llama 2](https://arxiv.org/abs/2307.09288), [Code Llama](https://arxiv.org/abs/2308.12950), [Falcon](https://huggingface.co/tiiuae) and [Mistral](https://arxiv.org/abs/2310.06825)
- support training of large models (70B Llama 2, 65B Llama 1, 34B Code Llama, 40B Falcon and Mistral) on commodity hardware on multiple nodes
- 3-way parallelism: tensor parallel, pipeline parallel and data parallel training (inherited from Megatron)
- full pretraining, finetuning and instruct tuning support
- Support for special tokens & tokenizers
- grouped-query attention (GQA) and multi-query attention (MQA)
- Rotary Position Embeddings (RoPE), RMS layer norm, Lima dropout
- [RoPE scaling](https://together.ai/blog/llama-2-7b-32k) for longer attention context support
- FlashAttention 2
- BF16 / FP16 training
- WandB integration
- Metrics support: Ease to add custom metrics to evaluate on the validation set while training
- Conversion to and from Hugging Face hub

# Documentation

Take a look at [the online documentation](https://epfllm.github.io/Megatron-LLM).

Alternatively, build the docs from source:
```
cd docs/
pip install -r requirements.txt
make html
```

# Example models trained with *Megatron-LLM*
70B Llama2: [meditron 70b](https://huggingface.co/epfl-llm/meditron-70b), [llama2-70b-oasst-sft-v10](https://huggingface.co/OpenAssistant/llama2-70b-oasst-sft-v10), 
40B Falcon: [falcon-40b-megacode2-oasst](https://huggingface.co/OpenAssistant/falcon-40b-megacode2-oasst), 
13B Code Llama: [codellama-13b-oasst-sft-v10](https://huggingface.co/OpenAssistant/codellama-13b-oasst-sft-v10), 
7B Llama2: [meditron 7b](https://huggingface.co/epfl-llm/meditron-7b), ...
(Let us know about yours!)

# Citation

If you use this software please cite it:
<pre>
@software{epfmgtrn,
  author       = {Alejandro Hernández Cano  and
                  Matteo Pagliardini  and
                  Andreas Köpf  and
                  Kyle Matoba  and
                  Amirkeivan Mohtashami  and
                  Xingyao Wang  and
                  Olivia Simin Fan  and
                  Axel Marmet  and
                  Deniz Bayazit  and
                  Igor Krawczuk  and
                  Zeming Chen  and
                  Francesco Salvi  and
                  Antoine Bosselut  and
                  Martin Jaggi},
  title        = {epfLLM Megatron-LLM},
  year         = 2023,
  url          = {https://github.com/epfLLM/Megatron-LLM}
}
</pre>
