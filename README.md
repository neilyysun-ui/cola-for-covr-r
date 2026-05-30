# CoVR-R Final Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python run_final.py \
  --test-json /path/to/test-set_no-labels.json \
  --video-dir /path/to/videos \
  --gemini-key /path/to/gemini_key \
  --output-json final_predictions.json
```

If WebVid ids cannot be resolved online, pass a candidate pool with `dataset_id`, `group`, `key`, and `file_path`:

```bash
python run_final.py \
  --test-json /path/to/test-set_no-labels.json \
  --video-dir /path/to/videos \
  --candidate-pool /path/to/candidate_pool.json \
  --gemini-key /path/to/gemini_key \
  --output-json final_predictions.json
```

The script writes only the JSON specified by `--output-json`.
