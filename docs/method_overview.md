# Method Overview

The pipeline separates candidate construction from final top-1 replacement.

## 1. Contact-Sheet Representation

Each video is represented by a compact contact sheet sampled from uniformly spaced frames. This keeps visual model calls consistent across short and long clips.

## 2. Reasoning and Text Seed

The reference video and edit text are converted into a target-side description. Candidate videos are described from their contact sheets. A local text ranker then creates the initial top-50 list.

## 3. Slot-Evidence Selection

The current candidate list is checked with structured visual evidence. This stage can update top-1 only when the challenger has clearly stronger evidence and lower risk.

## 4. Visual Candidate Route

DFN visual encoders retrieve visually compatible candidates from the gallery. This route is used to improve high-rank recall rather than directly replacing top-1.

## 5. Top-K Merge

The merged list preserves the current top-1, removes duplicates and the source video, and injects visual candidates into the top region.

## 6. Pairwise Reranking

The 1v1 stage compares the current top-1 against top challengers one at a time. A challenger is accepted only when it clearly satisfies the edit and the current top-1 has decisive visual errors.

## 7. Consensus Filter

Two structured VLM checks must agree on the same low-risk challenger before the final top-1 is changed. The remaining ranking is kept stable.
