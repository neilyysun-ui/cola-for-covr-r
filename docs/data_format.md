# Data Format

## Test Metadata

The runner expects a JSON list with separate `webvid` and `ss2` groups:

```json
[
  {
    "webvid": [
      {
        "id": 1,
        "video_source": "folder/example_source",
        "modification_text": "replace the action with a different target state"
      }
    ]
  },
  {
    "ss2": [
      {
        "id": 2,
        "video_source": 12345,
        "modification_text": "change the object interaction"
      }
    ]
  }
]
```

For `webvid`, the local source video key is inferred as the stem plus `.mp4`. For `ss2`, it is inferred as the stem plus `.webm`.

## Candidate Pool

The candidate pool is optional, but recommended when dataset ids are not recoverable from local filenames alone.

```json
[
  {
    "group": "webvid",
    "dataset_id": "folder/example_target",
    "key": "example_target.mp4",
    "file_path": "data/videos/example_target.mp4"
  },
  {
    "group": "ss2",
    "dataset_id": "67890",
    "key": "67890.webm",
    "file_path": "data/videos/67890.webm"
  }
]
```

Fields:

- `group`: `webvid` or `ss2`.
- `dataset_id`: id written into the final prediction list.
- `key`: local filename key.
- `file_path`: local video path. If omitted, the runner checks `--video-dir/key`.

## Output JSON

The runner writes:

```json
[
  {
    "webvid": [
      {
        "id": 1,
        "video_source": "folder/example_source",
        "video_target": ["folder/example_target"],
        "reasoning_trace": ["compact target reasoning and top prediction evidence"]
      }
    ]
  },
  {
    "ss2": [
      {
        "id": 2,
        "video_source": 12345,
        "video_target": ["67890"],
        "reasoning_trace": ["compact target reasoning and top prediction evidence"]
      }
    ]
  }
]
```

Each `video_target` list contains `--top-k` unique predictions.
