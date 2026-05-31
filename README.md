# CoVR-R Inference Pipeline

This repository contains a compact source-code package for the CoVR-R video retrieval pipeline. It is designed for final JSON generation from a test metadata file and a local video gallery.

Official data files, raw videos, generated predictions, contact-sheet caches, credentials, and large intermediate outputs are not included.

## Method Overview

The pipeline is training-free and runs entirely at inference time:

- Video candidates are converted into compact contact sheets.
- A vision-language model produces candidate descriptions and query-side target reasoning.
- A local text ranker builds the first top-50 ranking.
- A slot-evidence selector conservatively updates the current top-1 inside the existing candidate set.
- DFN visual encoders retrieve complementary visual candidates.
- The visual route injects candidates into the high-rank list while preserving the current top-1.
- A 1v1 reranker compares the current top-1 against top challengers.
- A structured consensus filter accepts only low-risk replacements.

The public package keeps the implementation runnable while omitting generated artifacts and private runtime files.

## Repository Structure

```text
.
├── run_final.py                  # Main end-to-end inference runner
├── requirements.txt              # Pip dependencies
├── environment.yml               # Conda environment
├── .env.example                  # Example secret configuration
├── configs/
│   └── run_full_inference.sh     # Generic shell recipe
├── docs/
│   ├── data_format.md            # Input, candidate pool, and output schema
│   └── method_overview.md        # High-level pipeline description
├── examples/
│   └── candidate_pool.example.json
└── tools/
    └── check_output.py           # Lightweight output validator
```

## Environment Setup

Conda:

```bash
conda env create -f environment.yml
conda activate covrr-final
```

Pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runner expects Python 3.11 or a compatible Python 3 environment with CUDA available for DFN image encoding. CPU execution is possible by passing `--device cpu`, but it is slower.

## API Key

Set the Gemini key in the environment:

```bash
export GEMINI_API_KEY=your_key_here
```

Alternatively, store the key in a private local file and pass it with `--gemini-key`.

Do not commit real keys or local path-specific configuration.

## Data Layout

A typical local layout is:

```text
data/
├── test-set_no-labels.json
├── videos/
│   ├── example_0001.webm
│   └── example_0002.mp4
└── candidate_pool.json
```

`candidate_pool.json` is optional when candidate ids can be resolved from the local video directory and public dataset file listing. For WebVid-style ids, using a candidate pool is recommended so folder-level ids are preserved.

See [docs/data_format.md](docs/data_format.md) for the expected schemas.

## Run Inference

Minimal command:

```bash
python run_final.py \
  --test-json data/test-set_no-labels.json \
  --video-dir data/videos \
  --output-json outputs/final_predictions.json
```

With a private key file and explicit candidate pool:

```bash
python run_final.py \
  --test-json data/test-set_no-labels.json \
  --video-dir data/videos \
  --candidate-pool data/candidate_pool.json \
  --gemini-key private/gemini_key.txt \
  --output-json outputs/final_predictions.json
```

The shell recipe in `configs/run_full_inference.sh` provides the same command through environment variables:

```bash
TEST_JSON=data/test-set_no-labels.json \
VIDEO_DIR=data/videos \
CANDIDATE_POOL=data/candidate_pool.json \
OUTPUT_JSON=outputs/final_predictions.json \
bash configs/run_full_inference.sh
```

## Useful Runtime Options

```text
--model                    Gemini model name
--device                   Torch device for visual encoders
--frames                   Number of sampled frames per contact sheet
--visual-models            Comma-separated visual encoder aliases
--gemini-workers           Number of concurrent API workers
--limit-queries            Small-run debugging limit
--limit-candidates         Small-gallery debugging limit
```

Defaults are set for the final pipeline. Adjust limits first when checking a new environment.

## Validate Output

After inference:

```bash
python tools/check_output.py \
  --output-json outputs/final_predictions.json \
  --expected-top-k 50
```

With the original test metadata:

```bash
python tools/check_output.py \
  --output-json outputs/final_predictions.json \
  --test-json data/test-set_no-labels.json \
  --expected-top-k 50
```

The checker verifies group keys, prediction length, duplicate ids, trace field type, and accidental inclusion of the source video when metadata is provided.

## Reproducibility Notes

- The runner writes only the JSON specified by `--output-json`.
- Contact sheets and intermediate state are created in a temporary directory.
- Generated outputs, credentials, local data, and caches are ignored by git.
- Remote model behavior can vary slightly across service updates.
