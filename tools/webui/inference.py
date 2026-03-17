import gc
import html
import re
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

DEFAULT_MAX_CHARS = 240
DEFAULT_MAX_LINES = 8
MIN_SPLIT_CHARS = 80
AUTO_MAX_NEW_TOKENS_MIN = 192
AUTO_MAX_NEW_TOKENS_MAX = 768


def inference_wrapper(
    text,
    reference_id,
    reference_audio_0,
    reference_text_0,
    reference_audio_1,
    reference_text_1,
    reference_audio_2,
    reference_text_2,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """

    references = get_reference_audio(
        [
            (reference_audio_0, reference_text_0),
            (reference_audio_1, reference_text_1),
            (reference_audio_2, reference_text_2),
        ]
    )

    request_args = dict(
        reference_id=reference_id if reference_id and not references else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
    )

    text = (text or "").strip()
    chunks = split_text_for_inference(text)
    queue = [(chunk, 0) for chunk in chunks]
    audio_segments = []
    sample_rate = None

    while queue:
        chunk_text, split_depth = queue.pop(0)
        req = ServeTTSRequest(
            text=chunk_text,
            max_new_tokens=resolve_max_new_tokens(
                chunk_text, request_args["max_new_tokens"]
            ),
            **{k: v for k, v in request_args.items() if k != "max_new_tokens"},
        )
        audio, error = run_inference_request(req, engine)

        if error is None:
            current_sample_rate, current_audio = audio
            if sample_rate is None:
                sample_rate = current_sample_rate
            audio_segments.append(current_audio)
            continue

        if is_oom_error(error):
            clear_cuda_memory()
            smaller_chunks = split_text_for_inference(
                chunk_text,
                max_chars=max(DEFAULT_MAX_CHARS // (2** (split_depth + 1)), MIN_SPLIT_CHARS),
                max_lines=max(DEFAULT_MAX_LINES // (2** (split_depth + 1)), 2),
            )
            if len(smaller_chunks) > 1:
                queue = [(piece, split_depth + 1) for piece in smaller_chunks] + queue
                continue

            return None, build_html_error_message(
                RuntimeError(
                    "入力が長すぎてGPUメモリが不足しました。より短い単位に分割して再実行してください。"
                )
            )

        return None, build_html_error_message(i18n(error))

    if not audio_segments or sample_rate is None:
        return None, i18n("No audio generated")

    return (sample_rate, np.concatenate(audio_segments, axis=0)), None


def get_reference_audio(reference_inputs: list[tuple[str, str]]) -> list:
    """
    Get the reference audio bytes.
    """

    references = []
    for reference_audio, reference_text in reference_inputs:
        if not reference_audio:
            continue

        with open(reference_audio, "rb") as audio_file:
            audio_bytes = audio_file.read()

        references.append(
            ServeReferenceAudio(audio=audio_bytes, text=reference_text or "")
        )

    return references


def run_inference_request(req: ServeTTSRequest, engine):
    try:
        for result in engine.inference(req):
            match result.code:
                case "final":
                    return result.audio, None
                case "error":
                    return None, result.error
                case _:
                    pass
    except Exception as error:
        return None, error

    return None, RuntimeError("No audio generated")


def split_text_for_inference(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, max_lines: int = DEFAULT_MAX_LINES
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= max_chars and text.count("\n") < max_lines:
        return [text]

    if "<|speaker:" in text:
        units = split_multispeaker_turns(text)
    else:
        units = [line.strip() for line in text.splitlines() if line.strip()]
        if len(units) <= 1:
            units = split_sentences(text)

    if not units:
        return [text]

    chunks = []
    current = []
    current_len = 0
    for unit in units:
        unit_len = len(unit)
        line_count = len(current)
        if current and (current_len + unit_len + 1 > max_chars or line_count >= max_lines):
            chunks.append("\n".join(current))
            current = [unit]
            current_len = unit_len
            continue

        current.append(unit)
        current_len += unit_len + (1 if current_len else 0)

    if current:
        chunks.append("\n".join(current))

    return chunks or [text]


def split_multispeaker_turns(text: str) -> list[str]:
    parts = re.split(r"(<\|speaker:\d+\|>)", text)
    turns = []
    current_speaker = None
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"<\|speaker:\d+\|>", part):
            current_speaker = part
            continue

        content = part.strip()
        if not content:
            continue
        turns.append(f"{current_speaker or '<|speaker:0|>'}{content}")

    return turns


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[。！？!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def resolve_max_new_tokens(text: str, max_new_tokens: int | float) -> int:
    if max_new_tokens and max_new_tokens > 0:
        return int(max_new_tokens)

    estimated = len(text.encode("utf-8")) * 2
    return max(AUTO_MAX_NEW_TOKENS_MIN, min(AUTO_MAX_NEW_TOKENS_MAX, estimated))


def is_oom_error(error: Any) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda out of memory" in message


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )
