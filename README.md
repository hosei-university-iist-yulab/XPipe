# XPipe

Explainable evaluation framework for multi-stage LLM pipelines. XPipe instruments
each stage of a Retrieval / Synthesis / Quality-Control pipeline, then computes
Shapley values from cooperative game theory to attribute the final quality
score to each stage. The framework was used to produce the results reported in
the ITU Kaleidoscope 2026 paper *XPipe: Explainable Multi-Stage LLM Pipeline
Evaluation via Causal Attribution for Telecommunications* (see Citation below).

## What it does

Given a query and a configuration of three stages

    Retrieval (S_r) -> Synthesis (S_s) -> Quality Control / Judge (S_j)

XPipe runs the pipeline with every subset of stages, scores each run with
ROUGE-L F1 against a reference answer, and estimates the Shapley value of each
stage via permutation sampling. Three additional metrics are computed on top
of the standard Shapley value:

- **Shapley-Latency Ratio** (quality contribution per millisecond)
- **Interaction Index** (synergy between stage pairs)
- **Convergence Bound** (variance bound on the sampling estimator)
- **Failure Shapley** and **Blame Score** for failure-conditioned attribution

## Repository layout

    xpipe/                   Python package
      theory.py              Shapley computation (permutation sampling)
      attribution.py         Factorial ablation / sweep helpers
      runners/rag.py         RAG runner (retrieval + synthesis + judge)
      retrievers.py          Lexical and dense retrievers
      llm_backends.py        Local HF + Ollama + Anthropic + Gemini backends
      metrics.py             ROUGE-L, Token F1, grounding, ECE
      pricing.py             Per-model cost table
      trace.py               JSONL trace / span logger
      outputs.py             CSV / LaTeX table writers
      plotting.py            Publication figures
      analysis/              Architecture diagram and statistical tests
    configs/
      experiment_rag.yaml    Default RAG pipeline config
      models.yaml            Model registry (5 local + 2 API)
    experiments/
      exp1_multidomain.py    Multi-domain configuration sweep
      exp2_attribution.py    Per-stage Shapley attribution
      exp3_cost_quality_latency.py
      exp4_dense_retrievers.py
    scripts/
      build_tbmp_dataset.py
      create_smartgrid_datasets.py
      prepare_datasets.py
      create_publication_plots.py
      cli_inspect.py
      plot_metrics.py
      run_*.sh               Batch experiment drivers
      readme.md              Per-script usage notes
    main.py                  Single-config pipeline entrypoint
    run_ablation.py          Stand-alone ablation runner
    test_all_components.py   Smoke test of retrievers, backends, metrics

## Install

Python 3.13.1 was used for the paper experiments. PyTorch with CUDA is required
to run the local HuggingFace backends on GPU.

    pip install -r requirements.txt

To use the optional API backends, install the relevant SDKs and set the
matching environment variables:

    pip install anthropic google-generativeai
    export ANTHROPIC_API_KEY=...
    export GOOGLE_API_KEY=...

## Quickstart

Run a single RAG pipeline with the default configuration:

    python main.py --config configs/experiment_rag.yaml

Run the full multi-domain sweep (Experiment 1 in the paper):

    python experiments/exp1_multidomain.py

Run the Shapley attribution analysis (Experiment 2 in the paper):

    python experiments/exp2_attribution.py

Both experiments write metrics to `output/exp1_multidomain/metrics/` and
`output/exp2_attribution/metrics/`, respectively. Publication-ready figures can
be regenerated with

    python scripts/create_publication_plots.py

See `scripts/readme.md` for batch drivers and parallel runners.

## Datasets

The five evaluation datasets used in the paper are not redistributed in this
repository. Each domain is sourced from a public origin:

- Network Troubleshooting: Stack Exchange Network Engineering
- Customer Support: Kaggle Customer Support on Twitter
- ITU Standards: ITU-T recommendations
- Legal Documents: USPTO Trademark Trial and Appeal Board Manual (TBMP)
- Energy / Electricity: Pecan Street Dataport

Helper scripts under `scripts/` (e.g. `build_tbmp_dataset.py`,
`create_smartgrid_datasets.py`, `prepare_datasets.py`) build the required
JSONL corpora from the public sources.

## Reproducing the paper results

The experiments in the paper were run on a single RTX 3090 (24 GB) with
Python 3.13.1 and PyTorch 2.7.1. The full Experiment 1 + Experiment 2 sweep
takes approximately six hours.

    bash scripts/run_all_experiments.sh
    python scripts/create_publication_plots.py

## Citation

If you use XPipe in your work, please cite:

    @inproceedings{messou2026xpipe,
      title     = {XPipe: Explainable Multi-Stage LLM Pipeline Evaluation
                   via Causal Attribution for Telecommunications},
      author    = {Messou, Franck Junior Aboya and Zhang, Shilong and
                   Wang, Weiyu and Yu, Tao and Liu, Tong and
                   Chen, Jinhua and Yu, Keping},
      booktitle = {Proc. ITU Kaleidoscope},
      year      = {2026}
    }

## License

BSD-3-Clause. See `LICENSE`.
