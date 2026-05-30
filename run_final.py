from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_REPO_ID = "orange-fox/CoVR-R"
THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Candidate:
    group: str
    key: str
    dataset_id: str
    file_path: str


@dataclass(frozen=True)
class QueryItem:
    group: str
    item_id: int
    video_source: str
    source_key: str
    modification_text: str

    @property
    def query_key(self) -> str:
        return f"{self.group}:{self.item_id}"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def source_to_key(group: str, source: Any) -> str:
    stem = Path(str(source)).name.split(".")[0]
    ext = ".mp4" if group == "webvid" else ".webm"
    return f"{stem}{ext}"


def source_to_dataset_id(group: str, source: Any) -> str:
    if group == "webvid":
        return str(source)
    return Path(str(source)).name.split(".")[0]


def detect_group(path_or_key: str) -> str:
    suffix = Path(path_or_key).suffix.lower()
    if suffix == ".mp4":
        return "webvid"
    if suffix == ".webm":
        return "ss2"
    raise ValueError(f"unsupported video suffix: {path_or_key}")


def load_items(path: Path, limit: int | None = None) -> list[QueryItem]:
    items: list[QueryItem] = []
    for section in read_json(path):
        for group, rows in section.items():
            for row in rows:
                item = QueryItem(
                    group=group,
                    item_id=int(row["id"]),
                    video_source=str(row["video_source"]),
                    source_key=source_to_key(group, row["video_source"]),
                    modification_text=str(row.get("modification_text", "")).strip(),
                )
                items.append(item)
                if limit is not None and len(items) >= limit:
                    return items
    return items


def candidate_from_repo_file(repo_file: str, video_dir: Path) -> Candidate | None:
    key = Path(repo_file).name
    suffix = Path(key).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return None
    local_path = video_dir / key
    if not local_path.exists():
        return None
    group = detect_group(key)
    if group == "webvid":
        parts = Path(repo_file).parts
        dataset_id = f"{parts[-2]}/{Path(key).stem}" if len(parts) >= 2 else Path(key).stem
    else:
        dataset_id = Path(key).stem
    return Candidate(group=group, key=key, dataset_id=dataset_id, file_path=str(local_path))


def load_candidates_from_pool(path: Path, video_dir: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for row in read_json(path):
        key = str(row.get("key") or Path(str(row.get("file_path", ""))).name)
        file_path = Path(str(row.get("file_path") or video_dir / key))
        if not file_path.exists():
            file_path = video_dir / key
        if not file_path.exists():
            continue
        group = str(row.get("group") or detect_group(key))
        dataset_id = str(row.get("dataset_id") or Path(key).stem)
        candidates.append(Candidate(group=group, key=key, dataset_id=dataset_id, file_path=str(file_path)))
    return sorted(candidates, key=lambda item: (item.group, item.key))


def load_candidates(video_dir: Path, candidate_pool: Path | None, repo_id: str, limit: int | None) -> list[Candidate]:
    if candidate_pool:
        candidates = load_candidates_from_pool(candidate_pool, video_dir)
    else:
        candidates = []
        try:
            from huggingface_hub import HfApi

            for repo_file in HfApi().list_repo_files(repo_id, repo_type="dataset"):
                candidate = candidate_from_repo_file(repo_file, video_dir)
                if candidate:
                    candidates.append(candidate)
        except Exception:
            candidates = []
        if not candidates:
            for path in sorted(video_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    group = detect_group(path.name)
                    candidates.append(Candidate(group=group, key=path.name, dataset_id=path.stem, file_path=str(path)))
    candidates = sorted(candidates, key=lambda item: (item.group, item.key))
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise RuntimeError("no candidate videos found")
    groups = {candidate.group for candidate in candidates}
    if "webvid" in groups and not any("/" in candidate.dataset_id for candidate in candidates if candidate.group == "webvid"):
        raise RuntimeError("webvid candidates need folder/id ids; pass --candidate-pool or enable Hugging Face access")
    return candidates


def candidate_lookup(candidates: list[Candidate]) -> dict[tuple[str, str], Candidate]:
    lookup: dict[tuple[str, str], Candidate] = {}
    for candidate in candidates:
        lookup[(candidate.group, candidate.dataset_id)] = candidate
        lookup[(candidate.group, candidate.key)] = candidate
        lookup[(candidate.group, Path(candidate.key).stem)] = candidate
    return lookup


def sample_indices(frame_count: int, frames: int) -> list[int]:
    if frame_count <= 0:
        return list(range(frames))
    if frame_count <= frames:
        return list(range(frame_count))
    return [int(round(value)) for value in __import__("numpy").linspace(0, frame_count - 1, frames)]


def read_video_frames(video_path: Path, frames: int) -> list[Any]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output = []
    for index in sample_indices(frame_count, frames):
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output.append(Image.fromarray(rgb))
    cap.release()
    if not output:
        raise RuntimeError(f"no frames decoded from video: {video_path}")
    return output


def fit_image(image: Any, width: int, height: int) -> Any:
    from PIL import Image

    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def draw_bar(image: Any, index: int, total: int) -> Any:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    bar_h = 8
    y0 = height - bar_h
    progress = 0.0 if total <= 1 else index / (total - 1)
    draw.rectangle([0, y0, width, height], fill=(30, 30, 30))
    draw.rectangle([0, y0, int(width * progress), height], fill=(80, 200, 120))
    return image


def make_contact_sheet(video_path: Path, sheet_path: Path, frames: int, cols: int, tile_width: int, tile_height: int) -> Path:
    from PIL import Image

    if sheet_path.exists():
        return sheet_path
    raw_frames = read_video_frames(video_path, frames)
    rows = math.ceil(len(raw_frames) / cols)
    sheet = Image.new("RGB", (cols * tile_width, rows * tile_height), "black")
    for idx, image in enumerate(raw_frames):
        tile = draw_bar(fit_image(image, tile_width, tile_height), idx, len(raw_frames))
        x = (idx % cols) * tile_width
        y = (idx // cols) * tile_height
        sheet.paste(tile, (x, y))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=88)
    return sheet_path


def extract_text(response: Any) -> str:
    parts = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?|```", "", text.strip(), flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(cleaned[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    raise RuntimeError("Gemini response did not contain a JSON object")


class GeminiClient:
    def __init__(self, args: argparse.Namespace) -> None:
        from google import genai

        api_key = args.gemini_api_key or ""
        if args.gemini_key:
            api_key = Path(args.gemini_key).read_text(encoding="utf-8").strip()
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("missing Gemini key; use --gemini-key or GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = args.model
        self.max_retries = args.max_retries
        self.upload_poll_seconds = args.upload_poll_seconds
        self.upload_timeout_seconds = args.upload_timeout_seconds
        self.sleep_seconds = args.sleep_seconds
        self.upload_cache: dict[str, Any] = {}
        self.lock = threading.Lock()

    def upload_file(self, path: Path) -> Any:
        key = str(path)
        with self.lock:
            cached = self.upload_cache.get(key)
        if cached is not None:
            return cached
        uploaded = self.client.files.upload(file=key)
        deadline = time.monotonic() + self.upload_timeout_seconds
        while uploaded.state.name == "PROCESSING":
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out while processing file: {path}")
            time.sleep(self.upload_poll_seconds)
            uploaded = self.client.files.get(name=uploaded.name)
        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Gemini file upload failed: {path}")
        with self.lock:
            self.upload_cache[key] = uploaded
        return uploaded

    def ask_json(self, prompt: str, image_paths: list[tuple[str, Path]], system_prompt: str, temperature: float, max_output_tokens: int, thinking_budget: int) -> dict[str, Any]:
        from google.genai import types

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=0.95,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinkingBudget=thinking_budget),
        )
        parts = [types.Part.from_text(text=prompt)]
        for label, path in image_paths:
            uploaded = self.upload_file(path)
            parts.append(types.Part.from_text(text=label))
            parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                text = extract_text(response)
                if not text:
                    raise RuntimeError("empty Gemini response")
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return extract_json_object(text)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(90.0, 5.0 * (2**attempt)) + random.uniform(0.0, 2.0))
        raise RuntimeError(f"Gemini request failed: {last_error}")


def get_gemini(args: argparse.Namespace) -> GeminiClient:
    client = getattr(THREAD_LOCAL, "gemini_client", None)
    if client is None:
        client = GeminiClient(args)
        THREAD_LOCAL.gemini_client = client
    return client


CANDIDATE_SYSTEM_PROMPT = """You are a video retrieval descriptor.
Describe only visual evidence visible in the contact sheet.
Return valid JSON only."""


QUERY_SYSTEM_PROMPT = """You are a composed video retrieval reasoner.
Infer the edited target video from the reference contact sheet and edit instruction.
Return valid JSON only."""


SLOT_SYSTEM_PROMPT = """You are a strict visual slot-evidence extractor for composed video retrieval.
Fill objective evidence for each candidate and do not choose by generic similarity.
Return valid JSON only."""


PAIRWISE_SYSTEM_PROMPT = """You are a conservative pairwise judge for composed video retrieval.
The current top1 is already strong. Choose the challenger only when it clearly matches the edit better and the current top1 has decisive visual errors.
Return valid JSON only."""


CONSENSUS_SLOT_SYSTEM_PROMPT = """You are a strict top5 slot verifier for composed video retrieval.
Extract target constraints and verify every candidate against hard visual evidence.
Return valid JSON only."""


CONSENSUS_RANK_SYSTEM_PROMPT = """You are a conservative top5 fatal-error ranking auditor for composed video retrieval.
Keep rank 1 unless another top5 candidate has fewer fatal errors and better target-specific evidence.
Return valid JSON only."""


def candidate_prompt(candidate: Candidate) -> str:
    return f"""Create a compact retrieval description for this candidate video.

Dataset group: {candidate.group}
Candidate id: {candidate.dataset_id}

Return this JSON:
{{
  "summary": "...",
  "objects": ["..."],
  "actions": ["..."],
  "temporal_phases": ["..."],
  "state_changes": ["..."],
  "scene": "...",
  "camera_framing": "...",
  "keywords": ["..."]
}}"""


def query_prompt(item: QueryItem) -> str:
    return f"""Given the reference video contact sheet and the modification text, infer the visual content of the target video.

Dataset group: {item.group}
Query id: {item.item_id}
Reference video id: {item.video_source}

Modification:
{item.modification_text}

Return this JSON:
{{
  "source_understanding": "...",
  "difference_proposal": "...",
  "target_description": "...",
  "search_terms": ["..."],
  "reasoning_trace": {{
    "states": "...",
    "actions": "...",
    "scene": "...",
    "camera": "...",
    "tempo": "..."
  }}
}}"""


def list_text(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if str(item).strip())
    if isinstance(value, dict):
        return "; ".join(f"{key}: {list_text(val)}" for key, val in value.items())
    if value is None:
        return ""
    return str(value)


def candidate_document(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    fields = [
        row.get("summary"),
        row.get("objects"),
        row.get("actions"),
        row.get("temporal_phases"),
        row.get("state_changes"),
        row.get("scene"),
        row.get("camera_framing"),
        row.get("keywords"),
    ]
    return "\n".join(list_text(field) for field in fields if list_text(field)).strip()


def query_document(row: dict[str, Any] | None, item: QueryItem) -> str:
    if not row:
        return item.modification_text
    fields = [
        row.get("target_description"),
        row.get("search_terms"),
        row.get("difference_proposal"),
        row.get("reasoning_trace"),
        item.modification_text,
    ]
    return "\n".join(list_text(field) for field in fields if list_text(field)).strip()


def reasoning_trace_text(reasoning: dict[str, Any] | None, top1: str, top1_caption: dict[str, Any] | None) -> str:
    if not reasoning:
        return f"Top-1 prediction: {top1}"
    pieces = [
        f"Source understanding: {list_text(reasoning.get('source_understanding'))}",
        f"Modification proposal: {list_text(reasoning.get('difference_proposal'))}",
        f"Target description: {list_text(reasoning.get('target_description'))}",
        f"Reasoning trace: {list_text(reasoning.get('reasoning_trace'))}",
        f"Search terms: {list_text(reasoning.get('search_terms'))}",
        f"Top-1 prediction: {top1}",
        f"Top-1 visual evidence: {candidate_document(top1_caption)}",
    ]
    return " ".join(piece for piece in pieces if piece.strip())


def sheet_path_for_candidate(candidate: Candidate, root: Path, args: argparse.Namespace) -> Path:
    return make_contact_sheet(
        Path(candidate.file_path),
        root / "candidates" / f"{candidate.key}.jpg",
        args.frames,
        args.contact_cols,
        args.tile_width,
        args.tile_height,
    )


def sheet_path_for_query(item: QueryItem, video_dir: Path, root: Path, args: argparse.Namespace) -> Path:
    return make_contact_sheet(
        video_dir / item.source_key,
        root / "queries" / f"{item.group}_{item.item_id}_{item.source_key}.jpg",
        args.frames,
        args.contact_cols,
        args.tile_width,
        args.tile_height,
    )


def parallel_map(items: list[Any], workers: int, fn: Any) -> list[Any]:
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def generate_candidate_captions(candidates: list[Candidate], sheet_root: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    def task(candidate: Candidate) -> tuple[str, dict[str, Any]]:
        sheet = sheet_path_for_candidate(candidate, sheet_root, args)
        payload = get_gemini(args).ask_json(
            candidate_prompt(candidate),
            [("candidate contact sheet", sheet)],
            CANDIDATE_SYSTEM_PROMPT,
            args.candidate_temperature,
            args.candidate_tokens,
            args.candidate_thinking_budget,
        )
        return candidate.key, payload

    return dict(parallel_map(candidates, args.gemini_workers, task))


def generate_query_reasoning(items: list[QueryItem], video_dir: Path, sheet_root: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    def task(item: QueryItem) -> tuple[str, dict[str, Any]]:
        sheet = sheet_path_for_query(item, video_dir, sheet_root, args)
        payload = get_gemini(args).ask_json(
            query_prompt(item),
            [("reference contact sheet", sheet)],
            QUERY_SYSTEM_PROMPT,
            args.query_temperature,
            args.query_tokens,
            args.query_thinking_budget,
        )
        return item.query_key, payload

    return dict(parallel_map(items, args.gemini_workers, task))


def vectorize_and_score(candidate_docs: list[str], query_docs: list[str]) -> Any:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    corpus = candidate_docs + query_docs
    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, norm="l2")
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True, norm="l2")
    matrix = hstack([word.fit_transform(corpus) * 0.7, char.fit_transform(corpus) * 0.3]).tocsr()
    matrix = normalize(matrix, norm="l2", copy=False)
    cand = matrix[: len(candidate_docs)]
    query = matrix[len(candidate_docs) :]
    return (query @ cand.T).toarray()


def build_text_seed(items: list[QueryItem], candidates: list[Candidate], captions: dict[str, dict[str, Any]], reasoning: dict[str, dict[str, Any]], top_k: int) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    by_group: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.key in captions:
            by_group.setdefault(candidate.group, []).append(candidate)
    for group, group_candidates in by_group.items():
        group_items = [item for item in items if item.group == group]
        if not group_items:
            continue
        candidate_docs = [candidate_document(captions.get(candidate.key)) for candidate in group_candidates]
        query_docs = [query_document(reasoning.get(item.query_key), item) for item in group_items]
        scores = vectorize_and_score(candidate_docs, query_docs)
        for query_idx, item in enumerate(group_items):
            source_id = source_to_dataset_id(item.group, item.video_source)
            ranked: list[str] = []
            for cand_idx in scores[query_idx].argsort()[::-1]:
                candidate = group_candidates[int(cand_idx)]
                if candidate.dataset_id == source_id or candidate.key == item.source_key:
                    continue
                ranked.append(candidate.dataset_id)
                if len(ranked) >= top_k:
                    break
            output[item.query_key] = ranked
    return output


def candidate_rows_for_ids(group: str, ids: list[str], candidates_by_id: dict[tuple[str, str], Candidate], captions: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset_id in ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        candidate = candidates_by_id.get((group, dataset_id))
        if not candidate:
            continue
        rows.append(
            {
                "rank": len(rows) + 1,
                "candidate_id": candidate.dataset_id,
                "key": candidate.key,
                "caption_text": candidate_document(captions.get(candidate.key))[:1200],
                "candidate": candidate,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def slot_selector_prompt(item: QueryItem, reasoning: dict[str, Any] | None, candidate_rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in candidate_rows:
        lines.append(f"Rank {row['rank']} candidate_id={row['candidate_id']}\n{row['caption_text']}")
    return f"""CoVR-R slot selection task.

Reference video id: {item.video_source}
Dataset group: {item.group}
Query id: {item.item_id}

Edit instruction:
{item.modification_text}

Target reasoning:
{query_document(reasoning, item)[:1800]}

Candidate list:
{chr(10).join(lines)}

Use the attached contact sheets: reference first, then candidates in rank order.

Return this JSON:
{{
  "candidate_slot_evidence": [
    {{
      "rank": 1,
      "candidate_id": "...",
      "primary_score": 0,
      "secondary_score": 0,
      "scene_score": 0,
      "action_score": 0,
      "forbidden_penalty": 0,
      "fatal_missing": ["..."],
      "fatal_violation": ["..."]
    }}
  ],
  "recommended_winner_rank": 1,
  "top1_replace_confidence": "low|medium|high",
  "replacement_risk": "low|medium|high"
}}"""


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def slot_score(evidence: dict[str, Any]) -> float:
    return (
        5.0 * number(evidence.get("primary_score"))
        + 2.0 * number(evidence.get("secondary_score"))
        + 2.0 * number(evidence.get("scene_score"))
        + 1.6 * number(evidence.get("action_score"))
        - 5.0 * number(evidence.get("forbidden_penalty"))
        - 6.0 * len(evidence.get("fatal_missing") or [])
        - 7.0 * len(evidence.get("fatal_violation") or [])
    )


def choose_slot_change(payload: dict[str, Any], candidate_rows: list[dict[str, Any]], margin: float) -> str | None:
    by_rank = {int_or(row.get("rank"), -1): row for row in payload.get("candidate_slot_evidence") or []}
    if 1 not in by_rank:
        return None
    current_score = slot_score(by_rank[1])
    challengers = [(rank, row, slot_score(row)) for rank, row in by_rank.items() if 1 < rank <= len(candidate_rows)]
    if not challengers:
        return None
    rank, row, score = max(challengers, key=lambda item: item[2])
    if score - current_score < margin:
        return None
    if row.get("fatal_missing") or row.get("fatal_violation"):
        return None
    if normalize_choice(payload.get("replacement_risk"), {"low", "medium", "high"}, "high") == "high":
        return None
    return str(candidate_rows[rank - 1]["candidate_id"])


def merge_unique(prefix: list[str], base: list[str], source_id: str, top_k: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in prefix + base:
        item = str(value)
        if not item or item == source_id or item in seen:
            continue
        output.append(item)
        seen.add(item)
        if len(output) >= top_k:
            break
    return output


def apply_slot_selection(items: list[QueryItem], rankings: dict[str, list[str]], candidates_by_id: dict[tuple[str, str], Candidate], captions: dict[str, dict[str, Any]], reasoning: dict[str, dict[str, Any]], video_dir: Path, sheet_root: Path, args: argparse.Namespace) -> dict[str, list[str]]:
    output = {key: list(value) for key, value in rankings.items()}

    def task(item: QueryItem) -> tuple[str, str | None]:
        current = rankings[item.query_key]
        rows = candidate_rows_for_ids(item.group, current, candidates_by_id, captions, args.slot_top_n)
        if len(rows) <= 1:
            return item.query_key, None
        images = [("reference contact sheet", sheet_path_for_query(item, video_dir, sheet_root, args))]
        for row in rows:
            images.append((f"rank {row['rank']} candidate", sheet_path_for_candidate(row["candidate"], sheet_root, args)))
        payload = get_gemini(args).ask_json(
            slot_selector_prompt(item, reasoning.get(item.query_key), rows),
            images,
            SLOT_SYSTEM_PROMPT,
            0.0,
            args.slot_tokens,
            args.slot_thinking_budget,
        )
        return item.query_key, choose_slot_change(payload, rows, args.slot_margin)

    for query_key, winner in parallel_map(items, args.gemini_workers, task):
        if winner:
            item = next(item for item in items if item.query_key == query_key)
            source_id = source_to_dataset_id(item.group, item.video_source)
            output[query_key] = merge_unique([winner], output[query_key], source_id, args.top_k)
    return output


def load_open_clip_model(alias: str, device: str) -> tuple[Any, Any, Any, Any]:
    import open_clip
    import torch

    specs = {
        "dfn_h": ("ViT-H-14-378-quickgelu", "dfn5b"),
        "dfn_l": ("ViT-L-14-quickgelu", "dfn2b"),
    }
    if alias not in specs:
        raise ValueError(f"unknown visual model: {alias}")
    model_name, pretrained = specs[alias]
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, torch


def encode_images(model: Any, preprocess: Any, paths: list[Path], torch: Any, device: str, batch_size: int) -> Any:
    from PIL import Image

    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            batch = torch.stack(images).to(device)
            feat = model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            outputs.append(feat.cpu())
    return torch.cat(outputs, dim=0).numpy()


def encode_texts(model: Any, tokenizer: Any, texts: list[str], torch: Any, device: str, batch_size: int) -> Any:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            tokens = tokenizer(batch_texts).to(device)
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            outputs.append(feat.cpu())
    return torch.cat(outputs, dim=0).numpy()


def visual_prompts(item: QueryItem, reasoning: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not reasoning:
        return [(item.modification_text, 1.0)]
    prompts = [
        (str(reasoning.get("target_description") or ""), 1.0),
        (list_text(reasoning.get("search_terms")), 0.7),
        (item.modification_text, 0.5),
    ]
    return [(text, weight) for text, weight in prompts if text.strip()]


def visual_rankings(items: list[QueryItem], candidates: list[Candidate], reasoning: dict[str, dict[str, Any]], sheet_root: Path, args: argparse.Namespace) -> dict[str, list[str]]:
    import numpy as np

    candidate_paths = [sheet_path_for_candidate(candidate, sheet_root, args) for candidate in candidates]
    by_group: dict[str, list[int]] = {}
    for idx, candidate in enumerate(candidates):
        by_group.setdefault(candidate.group, []).append(idx)
    combined: dict[str, np.ndarray] = {}
    for alias in args.visual_models.split(","):
        alias = alias.strip()
        if not alias:
            continue
        model, preprocess, tokenizer, torch = load_open_clip_model(alias, args.device)
        image_embeddings = encode_images(model, preprocess, candidate_paths, torch, args.device, args.image_batch_size)
        for item in items:
            prompt_rows = visual_prompts(item, reasoning.get(item.query_key))
            texts = [row[0] for row in prompt_rows]
            weights = np.array([row[1] for row in prompt_rows], dtype="float32")
            text_embeddings = encode_texts(model, tokenizer, texts, torch, args.device, args.text_batch_size)
            scores = (text_embeddings @ image_embeddings.T).astype("float32")
            score = (scores * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-6)
            if item.query_key not in combined:
                combined[item.query_key] = score
            else:
                combined[item.query_key] += score
    rankings: dict[str, list[str]] = {}
    model_count = max(1, len([part for part in args.visual_models.split(",") if part.strip()]))
    for item in items:
        score = combined[item.query_key] / model_count
        group_indices = np.array(by_group.get(item.group, []), dtype="int64")
        order = group_indices[np.argsort(score[group_indices])[::-1]]
        source_id = source_to_dataset_id(item.group, item.video_source)
        ranked: list[str] = []
        for idx in order[: args.visual_top_k]:
            candidate = candidates[int(idx)]
            if candidate.dataset_id == source_id or candidate.key == item.source_key:
                continue
            ranked.append(candidate.dataset_id)
        rankings[item.query_key] = ranked
    return rankings


def apply_visual_injection(items: list[QueryItem], current: dict[str, list[str]], visual: dict[str, list[str]], args: argparse.Namespace) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for item in items:
        base = current[item.query_key]
        source_id = source_to_dataset_id(item.group, item.video_source)
        injected = [base[0]] + visual.get(item.query_key, [])[: args.visual_inject_top] + base[1:]
        output[item.query_key] = merge_unique(injected, visual.get(item.query_key, []), source_id, args.top_k)
    return output


def pairwise_prompt(item: QueryItem, reasoning: dict[str, Any] | None, current_id: str, challenger_id: str, current_caption: dict[str, Any] | None, challenger_caption: dict[str, Any] | None) -> str:
    return f"""CoVR-R 1v1 top1 replacement task.

Reference video id: {item.video_source}
Dataset group: {item.group}
Query id: {item.item_id}

Edit instruction:
{item.modification_text}

Target reasoning:
{query_document(reasoning, item)[:1800]}

Current top1 A candidate_id={current_id}
{candidate_document(current_caption)[:900]}

Challenger B candidate_id={challenger_id}
{candidate_document(challenger_caption)[:900]}

Use the attached contact sheets in this order: reference, current A, challenger B.

Return this JSON:
{{
  "current_A": {{
    "target_match_score": 0,
    "fatal_errors": ["..."]
  }},
  "challenger_B": {{
    "target_match_score": 0,
    "fatal_errors": ["..."]
  }},
  "winner": "A",
  "challenger_better": false,
  "confidence": "low|medium|high",
  "replacement_risk": "low|medium|high"
}}"""


def evidence_score(evidence: dict[str, Any]) -> float:
    return number(evidence.get("target_match_score")) - 7.0 * len(evidence.get("fatal_errors") or [])


def apply_pairwise_rerank(items: list[QueryItem], current: dict[str, list[str]], candidates_by_id: dict[tuple[str, str], Candidate], captions: dict[str, dict[str, Any]], reasoning: dict[str, dict[str, Any]], video_dir: Path, sheet_root: Path, args: argparse.Namespace) -> dict[str, list[str]]:
    output = {key: list(value) for key, value in current.items()}
    tasks: list[tuple[QueryItem, str]] = []
    for item in items:
        for challenger_id in current[item.query_key][1 : args.pairwise_top_n]:
            tasks.append((item, challenger_id))

    def task(row: tuple[QueryItem, str]) -> tuple[str, str | None, float]:
        item, challenger_id = row
        ranking = current[item.query_key]
        current_id = ranking[0]
        current_candidate = candidates_by_id.get((item.group, current_id))
        challenger_candidate = candidates_by_id.get((item.group, challenger_id))
        if not current_candidate or not challenger_candidate:
            return item.query_key, None, 0.0
        images = [
            ("reference contact sheet", sheet_path_for_query(item, video_dir, sheet_root, args)),
            ("current A contact sheet", sheet_path_for_candidate(current_candidate, sheet_root, args)),
            ("challenger B contact sheet", sheet_path_for_candidate(challenger_candidate, sheet_root, args)),
        ]
        payload = get_gemini(args).ask_json(
            pairwise_prompt(
                item,
                reasoning.get(item.query_key),
                current_id,
                challenger_id,
                captions.get(current_candidate.key),
                captions.get(challenger_candidate.key),
            ),
            images,
            PAIRWISE_SYSTEM_PROMPT,
            0.0,
            args.pairwise_tokens,
            args.pairwise_thinking_budget,
        )
        current_ev = payload.get("current_A") or {}
        challenger_ev = payload.get("challenger_B") or {}
        margin = evidence_score(challenger_ev) - evidence_score(current_ev)
        challenger_wins = payload.get("challenger_better") is True or str(payload.get("winner") or "").strip().upper() == "B"
        confidence = normalize_choice(payload.get("confidence"), {"low", "medium", "high"}, "low")
        risk = normalize_choice(payload.get("replacement_risk"), {"low", "medium", "high"}, "high")
        current_fatal = len(current_ev.get("fatal_errors") or [])
        challenger_fatal = len(challenger_ev.get("fatal_errors") or [])
        accept = challenger_wins and confidence == "high" and risk == "low" and margin > args.pairwise_margin and current_fatal >= 1 and challenger_fatal == 0
        return item.query_key, challenger_id if accept else None, margin

    winners: dict[str, tuple[str, float]] = {}
    for query_key, winner, margin in parallel_map(tasks, args.gemini_workers, task):
        if winner and (query_key not in winners or margin > winners[query_key][1]):
            winners[query_key] = (winner, margin)
    item_by_key = {item.query_key: item for item in items}
    for query_key, (winner, _) in winners.items():
        item = item_by_key[query_key]
        output[query_key] = merge_unique([winner], output[query_key], source_to_dataset_id(item.group, item.video_source), args.top_k)
    return output


def consensus_slot_prompt(item: QueryItem, reasoning: dict[str, Any] | None, rows: list[dict[str, Any]]) -> str:
    lines = [f"Rank {row['rank']} candidate_id={row['candidate_id']}\n{row['caption_text']}" for row in rows]
    return f"""CoVR-R top5 slot verification task.

Reference video id: {item.video_source}
Dataset group: {item.group}
Query id: {item.item_id}

Edit instruction:
{item.modification_text}

Target reasoning:
{query_document(reasoning, item)[:1800]}

Candidate list:
{chr(10).join(lines)}

Return this JSON:
{{
  "candidate_slot_evidence": [
    {{
      "rank": 1,
      "candidate_id": "...",
      "primary_score": 0,
      "action_state_score": 0,
      "scene_camera_score": 0,
      "preserve_score": 0,
      "forbidden_penalty": 0,
      "fatal_missing": ["..."],
      "fatal_violation": ["..."],
      "uncertain_critical_slots": ["..."]
    }}
  ],
  "recommended_winner_rank": 1,
  "top1_replace_confidence": "low|medium|high",
  "replacement_risk": "low|medium|high"
}}"""


def consensus_rank_prompt(item: QueryItem, reasoning: dict[str, Any] | None, rows: list[dict[str, Any]]) -> str:
    lines = [f"Rank {row['rank']} candidate_id={row['candidate_id']}\n{row['caption_text']}" for row in rows]
    return f"""CoVR-R top5 fatal-error ranking task.

Reference video id: {item.video_source}
Dataset group: {item.group}
Query id: {item.item_id}

Edit instruction:
{item.modification_text}

Target reasoning:
{query_document(reasoning, item)[:1800]}

Candidate list:
{chr(10).join(lines)}

Return this JSON:
{{
  "candidate_assessments": [
    {{
      "rank": 1,
      "candidate_id": "...",
      "target_match_score": 0,
      "fatal_errors": ["..."],
      "unsafe_as_top1": false
    }}
  ],
  "winner_rank": 1,
  "should_replace_top1": false,
  "confidence": "low|medium|high",
  "replacement_risk": "low|medium|high",
  "current_top1_fatal_errors": ["..."],
  "winner_fatal_errors": ["..."]
}}"""


def choose_consensus_slot(payload: dict[str, Any], margin: float, top_n: int) -> dict[str, Any]:
    by_rank = {int_or(row.get("rank"), -1): row for row in payload.get("candidate_slot_evidence") or []}
    if 1 not in by_rank:
        return {"accept": False}
    current = by_rank[1]
    current_score = slot_score(current)
    challengers = [(rank, row, slot_score(row)) for rank, row in by_rank.items() if 1 < rank <= top_n]
    if not challengers:
        return {"accept": False}
    rank, row, score = max(challengers, key=lambda item: item[2])
    current_fatal = len(current.get("fatal_missing") or []) + len(current.get("fatal_violation") or [])
    winner_fatal = len(row.get("fatal_missing") or []) + len(row.get("fatal_violation") or [])
    accept = (
        score - current_score >= margin
        and current_fatal >= 1
        and winner_fatal == 0
        and not row.get("uncertain_critical_slots")
        and normalize_choice(payload.get("top1_replace_confidence"), {"low", "medium", "high"}, "low") == "high"
        and normalize_choice(payload.get("replacement_risk"), {"low", "medium", "high"}, "high") == "low"
    )
    return {"accept": accept, "winner_rank": rank, "margin": score - current_score, "current_fatal": current_fatal, "winner_fatal": winner_fatal}


def choose_consensus_rank(payload: dict[str, Any], top_n: int) -> dict[str, Any]:
    winner_rank = int_or(payload.get("winner_rank"), -1)
    assessments = {int_or(row.get("rank"), -1): row for row in payload.get("candidate_assessments") or []}
    winner = assessments.get(winner_rank, {})
    current = assessments.get(1, {})
    current_fatal = len(payload.get("current_top1_fatal_errors") or current.get("fatal_errors") or [])
    winner_fatal = len(payload.get("winner_fatal_errors") or winner.get("fatal_errors") or [])
    accept = (
        1 < winner_rank <= top_n
        and payload.get("should_replace_top1") is True
        and normalize_choice(payload.get("confidence"), {"low", "medium", "high"}, "low") == "high"
        and normalize_choice(payload.get("replacement_risk"), {"low", "medium", "high"}, "high") == "low"
        and current_fatal >= 1
        and winner_fatal == 0
        and winner.get("unsafe_as_top1") is not True
    )
    return {"accept": accept, "winner_rank": winner_rank, "current_fatal": current_fatal, "winner_fatal": winner_fatal}


def apply_consensus(items: list[QueryItem], current: dict[str, list[str]], candidates_by_id: dict[tuple[str, str], Candidate], captions: dict[str, dict[str, Any]], reasoning: dict[str, dict[str, Any]], video_dir: Path, sheet_root: Path, args: argparse.Namespace) -> dict[str, list[str]]:
    output = {key: list(value) for key, value in current.items()}

    def task(item: QueryItem) -> tuple[str, str | None]:
        rows = candidate_rows_for_ids(item.group, current[item.query_key], candidates_by_id, captions, args.consensus_top_n)
        if len(rows) <= 1:
            return item.query_key, None
        images = [("reference contact sheet", sheet_path_for_query(item, video_dir, sheet_root, args))]
        for row in rows:
            images.append((f"rank {row['rank']} candidate", sheet_path_for_candidate(row["candidate"], sheet_root, args)))
        client = get_gemini(args)
        slot_payload = client.ask_json(
            consensus_slot_prompt(item, reasoning.get(item.query_key), rows),
            images,
            CONSENSUS_SLOT_SYSTEM_PROMPT,
            0.0,
            args.consensus_tokens,
            args.consensus_thinking_budget,
        )
        rank_payload = client.ask_json(
            consensus_rank_prompt(item, reasoning.get(item.query_key), rows),
            images,
            CONSENSUS_RANK_SYSTEM_PROMPT,
            0.0,
            args.consensus_tokens,
            args.consensus_thinking_budget,
        )
        slot = choose_consensus_slot(slot_payload, args.consensus_margin, args.consensus_top_n)
        rank = choose_consensus_rank(rank_payload, args.consensus_top_n)
        if not slot.get("accept") or not rank.get("accept"):
            return item.query_key, None
        if slot.get("winner_rank") != rank.get("winner_rank"):
            return item.query_key, None
        winner_rank = int(slot["winner_rank"])
        return item.query_key, str(rows[winner_rank - 1]["candidate_id"])

    item_by_key = {item.query_key: item for item in items}
    for query_key, winner in parallel_map(items, args.gemini_workers, task):
        if winner:
            item = item_by_key[query_key]
            output[query_key] = merge_unique([winner], output[query_key], source_to_dataset_id(item.group, item.video_source), args.top_k)
    return output


def output_json(items: list[QueryItem], rankings: dict[str, list[str]], candidates_by_id: dict[tuple[str, str], Candidate], captions: dict[str, dict[str, Any]], reasoning: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"webvid": [], "ss2": []}
    for item in items:
        ranking = rankings[item.query_key][:top_k]
        if len(ranking) != top_k:
            raise RuntimeError(f"{item.query_key} has {len(ranking)} predictions, expected {top_k}")
        source_id = source_to_dataset_id(item.group, item.video_source)
        if source_id in ranking:
            raise RuntimeError(f"{item.query_key} includes source video")
        if len(set(ranking)) != len(ranking):
            raise RuntimeError(f"{item.query_key} has duplicate predictions")
        top_candidate = candidates_by_id.get((item.group, ranking[0]))
        top_caption = captions.get(top_candidate.key) if top_candidate else None
        grouped[item.group].append(
            {
                "id": item.item_id,
                "video_source": item.video_source if item.group == "webvid" else int(item.video_source),
                "video_target": ranking,
                "reasoning_trace": [reasoning_trace_text(reasoning.get(item.query_key), ranking[0], top_caption)],
            }
        )
    return [{"webvid": grouped["webvid"]}, {"ss2": grouped["ss2"]}]


def run(args: argparse.Namespace) -> None:
    test_json = Path(args.test_json)
    video_dir = Path(args.video_dir)
    output_path = Path(args.output_json)
    candidate_pool = Path(args.candidate_pool) if args.candidate_pool else None
    items = load_items(test_json, args.limit_queries)
    candidates = load_candidates(video_dir, candidate_pool, args.repo_id, args.limit_candidates)
    candidates_by_id = candidate_lookup(candidates)
    with tempfile.TemporaryDirectory(prefix="covrr_final_") as tmp:
        sheet_root = Path(tmp) / "contact_sheets"
        captions = generate_candidate_captions(candidates, sheet_root, args)
        reasoning = generate_query_reasoning(items, video_dir, sheet_root, args)
        text_seed = build_text_seed(items, candidates, captions, reasoning, args.top_k)
        slot_ranked = apply_slot_selection(items, text_seed, candidates_by_id, captions, reasoning, video_dir, sheet_root, args)
        visual = visual_rankings(items, candidates, reasoning, sheet_root, args)
        visual_ranked = apply_visual_injection(items, slot_ranked, visual, args)
        pairwise_ranked = apply_pairwise_rerank(items, visual_ranked, candidates_by_id, captions, reasoning, video_dir, sheet_root, args)
        final_ranked = apply_consensus(items, pairwise_ranked, candidates_by_id, captions, reasoning, video_dir, sheet_root, args)
        write_json(output_path, output_json(items, final_ranked, candidates_by_id, captions, reasoning, args.top_k))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gemini-key", default="")
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--candidate-pool", default="")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visual-models", default="dfn_h,dfn_l")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--contact-cols", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=220)
    parser.add_argument("--gemini-workers", type=int, default=4)
    parser.add_argument("--candidate-temperature", type=float, default=0.1)
    parser.add_argument("--query-temperature", type=float, default=0.2)
    parser.add_argument("--candidate-tokens", type=int, default=900)
    parser.add_argument("--query-tokens", type=int, default=1400)
    parser.add_argument("--candidate-thinking-budget", type=int, default=512)
    parser.add_argument("--query-thinking-budget", type=int, default=1600)
    parser.add_argument("--slot-top-n", type=int, default=5)
    parser.add_argument("--slot-margin", type=float, default=5.0)
    parser.add_argument("--slot-tokens", type=int, default=6000)
    parser.add_argument("--slot-thinking-budget", type=int, default=2048)
    parser.add_argument("--visual-top-k", type=int, default=100)
    parser.add_argument("--visual-inject-top", type=int, default=14)
    parser.add_argument("--image-batch-size", type=int, default=96)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--pairwise-top-n", type=int, default=5)
    parser.add_argument("--pairwise-margin", type=float, default=5.0)
    parser.add_argument("--pairwise-tokens", type=int, default=5000)
    parser.add_argument("--pairwise-thinking-budget", type=int, default=1024)
    parser.add_argument("--consensus-top-n", type=int, default=5)
    parser.add_argument("--consensus-margin", type=float, default=8.0)
    parser.add_argument("--consensus-tokens", type=int, default=7000)
    parser.add_argument("--consensus-thinking-budget", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--upload-poll-seconds", type=float, default=2.0)
    parser.add_argument("--upload-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
