---
license: {LICENSE_TAG}
base_model: {BASE_MODEL}
tags:
- interpretability
- activation-decoding
- nla
---

{BUILT_WITH_BANNER}

# {DISPLAY_NAME}

The **{ROLE_DESC}** half of a Natural Language Autoencoder (NLA) pair,
fine-tuned from [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}). The
other half is [`{PAIR_REPO}`](https://huggingface.co/{PAIR_REPO}); both are
released together and are intended to be used as a pair.

NLA pairs are interpretability tools: the AV (activation verbalizer) maps a
hidden-state vector to a natural-language description; the AR (activation
reconstructor) maps that description back to a vector. Together they let you
read out what a residual-stream activation "means" and measure how much of it
the description captured. **These checkpoints are not useful as general-purpose
language models** — the fine-tuning repurposes them entirely for activation
decoding.

- 📄 Paper: [Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations](https://transformer-circuits.pub/2026/nla/index.html)
- Inference code + worked examples: [`kitft/nla-inference`](https://github.com/kitft/nla-inference)
- Training code: [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders)
- Extraction layer: residual stream output of block **{LAYER_IDX}**
- In-distribution fve_nrm: **{TRAINING_FVE}** (training set, 50/50 WildChat + Ultra-FineWeb)

## Usage

See the [nla-inference README](https://github.com/kitft/nla-inference) for the
full recipe (SGLang launch, `NLAClient`/`NLACritic`, embedding-injection
details).

## Citation

```bibtex
@article{{frasertaliente2026nla,
  author  = {{Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and Mossing, Dan and Lu, Christina and Bogdan, Paul C. and Ameisen, Emmanuel and Chen, James and Kishylau, Dzmitry and Pearce, Adam and Tarng, Julius and Wu, Alex and Wu, Jeff and Zhang, Yang and Ziegler, Daniel M. and Hubinger, Evan and Batson, Joshua and Lindsey, Jack and Zimmerman, Samuel and Marks, Samuel}},
  title   = {{Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations}},
  journal = {{Transformer Circuits Thread}},
  year    = {{2026}},
  url     = {{https://transformer-circuits.pub/2026/nla/index.html}}
}}
```

## License & use restrictions

{LICENSE_STANZA}

## Training data attribution

The fine-tuning data was derived from two public datasets:

- **WildChat-1M** ([allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)).
  Contains information from WildChat-1M which is made available under the
  [ODC Attribution License](https://opendatacommons.org/licenses/by/1-0/).
- **Ultra-FineWeb** ([openbmb/Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb),
  Apache-2.0), a filtered derivative of
  [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
  (ODC-BY).
