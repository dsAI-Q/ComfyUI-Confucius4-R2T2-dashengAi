# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# ComfyUI custom nodes for Confucius4-R2T2 — a multilingual streaming / one-shot
# ASR model built on the Qwen3-ASR architecture (NetEase Youdao).
#
# Node list:
#   * R2T2ModelLoader          — load the model with the transformers or vLLM backend.
#   * R2T2Transcribe           — one-shot (non-streaming) transcription.
#   * R2T2StreamingTranscribe  — streaming transcription (vLLM backend only).
#
# Heavy imports (torch / librosa / qwen_asr / r2t2) are deferred to the first
# node call so that ComfyUI startup is not slowed down.

import os
import re
import string

import numpy as np

# -----------------------------------------------------------------------------
# Model cache
# -----------------------------------------------------------------------------
_MODEL_CACHE = {}

# Default location for R2T2 checkpoints inside the ComfyUI models tree.
_MODEL_SUBDIR = "r2t2"


def _get_model_class():
    """Lazily import the R2T2ASRModel class (vendored `r2t2` package)."""
    from r2t2 import R2T2ASRModel  # noqa: PLC0415
    return R2T2ASRModel


def _clear_cache():
    _MODEL_CACHE.clear()


# -----------------------------------------------------------------------------
# Model path resolution / download (Hugging Face, stored under ComfyUI/models)
# -----------------------------------------------------------------------------
def _models_root() -> str:
    """Return the ComfyUI `models` directory.

    Resolution order:
      1. ComfyUI's own `folder_paths.models_dir` (preferred, in-process),
      2. the COMFYUI_MODELS_DIR environment variable,
      3. a `ComfyUI/models` directory found relative to the current working dir,
      4. the current working directory as a last resort.
    """
    try:
        import folder_paths  # noqa: PLC0415
        root = folder_paths.models_dir
        if root and os.path.isdir(root):
            return root
    except Exception:  # noqa: BLE001
        pass

    env = os.environ.get("COMFYUI_MODELS_DIR", "").strip()
    if env and os.path.isdir(env):
        return env

    # Walk up from the CWD to find a ComfyUI/models directory.
    cwd = os.path.abspath(os.getcwd())
    probe = cwd
    for _ in range(5):
        candidate = os.path.join(probe, "ComfyUI", "models")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return cwd


def _local_model_dir(repo_id: str) -> str:
    """Map an HF repo id to ComfyUI/models/r2t2/<org>/<repo>."""
    safe = repo_id.strip().rstrip("/")
    return os.path.join(_models_root(), _MODEL_SUBDIR, safe.replace("/", os.sep))


def _repo_is_available(repo_dir: str) -> bool:
    """A checkpoint is considered present when its config.json exists on disk."""
    return os.path.isfile(os.path.join(repo_dir, "config.json"))


# Hugging Face mirror that works without a proxy in mainland China.
_HF_MIRROR = "https://hf-mirror.com"


def _download_model_snapshot(repo_id: str, repo_dir: str, endpoint=None) -> None:
    """Download a full HF repo snapshot to `repo_dir`."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    snapshot_download(
        repo_id=repo_id,
        local_dir=repo_dir,
        local_dir_use_symlinks=False,
        endpoint=endpoint,
    )


def _manual_download_hint(repo_id: str, repo_dir: str) -> str:
    return (
        f"Model '{repo_id}' could not be downloaded automatically. "
        "Please download it manually (see README):\n"
        f"  # option 1: direct / with proxy (overseas network)\n"
        f"  huggingface-cli download {repo_id} --local-dir \"{repo_dir}\"\n"
        f"  # option 2: mainland mirror, no proxy needed (recommended)\n"
        f"  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download {repo_id} --local-dir \"{repo_dir}\""
    )


def _ensure_model(model_path: str, auto_download: bool = True) -> str:
    """
    Resolve `model_path` to a real local path.

    * Local path (exists on disk)        -> used as-is.
    * Hugging Face repo id               -> downloaded (if auto_download) into
                                           ComfyUI/models/r2t2/<org>/<repo>,
                                           or an error tells the user how to
                                           download it manually.
    """
    mp = model_path.strip()
    if not mp:
        raise ValueError("model_path must not be empty.")

    if os.path.isdir(mp):
        return mp

    # Treat it as an HF repo id.
    repo_dir = _local_model_dir(mp)
    if _repo_is_available(repo_dir):
        return repo_dir

    if not auto_download:
        raise RuntimeError(
            f"Model '{mp}' is not found locally at:\n  {repo_dir}\n\n"
            + _manual_download_hint(mp, repo_dir)
        )

    print(f"[Confucius4-R2T2] Downloading model '{mp}' to:\n  {repo_dir}")
    last_err = None
    try:
        # 1) Default endpoint (respects HF_ENDPOINT / proxy env of the process).
        _download_model_snapshot(mp, repo_dir, endpoint=None)
    except Exception as e:  # noqa: BLE001
        last_err = e
        # 2) Fallback: hf-mirror.com works without a proxy in mainland networks.
        if os.environ.get("HF_ENDPOINT", "").strip() != _HF_MIRROR:
            print(f"[Confucius4-R2T2] Default endpoint failed ({type(e).__name__}). "
                  f"Retrying via mirror {_HF_MIRROR} ...")
            try:
                _download_model_snapshot(mp, repo_dir, endpoint=_HF_MIRROR)
            except Exception as e2:  # noqa: BLE001
                raise RuntimeError(
                    _manual_download_hint(mp, repo_dir) + f"\n\nErrors: {e}\n{e2}"
                ) from e2
        else:
            raise RuntimeError(_manual_download_hint(mp, repo_dir) + f"\n\nError: {e}") from e

    if not _repo_is_available(repo_dir):
        raise RuntimeError(f"Model download finished but config.json is missing in {repo_dir}.")
    return repo_dir


# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------
def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample (linear interp) an arbitrary-rate mono waveform to 16 kHz float32."""
    wav = np.asarray(wav, dtype=np.float32)
    if sr == 16000:
        return wav
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _load_audio_from_path(path: str) -> np.ndarray:
    """Load an audio file from disk and return 16 kHz mono float32 PCM."""
    if not path or not path.strip():
        raise ValueError("audio_path is empty. Provide an audio file path or connect an AUDIO input.")
    if not os.path.exists(path.strip()):
        raise FileNotFoundError(f"Audio file not found: {path.strip()}")
    import librosa  # noqa: PLC0415
    audio, sr = librosa.load(path.strip(), sr=None, mono=True)
    return _resample_to_16k(audio, int(sr))


def _audio_input_to_16k(audio=None, audio_path: str = "") -> np.ndarray:
    """
    Normalize a ComfyUI `AUDIO` dict or a file path into 16 kHz mono float32 PCM.

    Supported AUDIO dict shape (VideoHelperSuite & friends):
        {"waveform": torch.Tensor [batch, channels, samples], "sample_rate": int}
    """
    if audio is not None:
        try:
            import torch  # noqa: PLC0415
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
        except (KeyError, TypeError) as e:
            raise ValueError(
                "Unsupported AUDIO input. Expected a dict with 'waveform' and 'sample_rate'."
            ) from e

        if isinstance(waveform, torch.Tensor):
            wav = waveform.detach().float().cpu().numpy()
        else:
            wav = np.asarray(waveform, dtype=np.float32)

        # [batch, channels, samples] -> mono
        if wav.ndim == 3:
            wav = wav[0]
        if wav.ndim == 2:
            wav = wav.mean(axis=0)
        return _resample_to_16k(wav, sr)

    return _load_audio_from_path(audio_path)


def _language_param(language: str):
    """Map 'auto'/'empty' to None (auto-detect), otherwise pass through."""
    if language is None:
        return None
    lang = str(language).strip()
    if lang == "" or lang.lower() == "auto":
        return None
    return lang


# -----------------------------------------------------------------------------
# Streaming logic (mirrors the upstream run_streaming() example)
# -----------------------------------------------------------------------------
def _remove_punctuation(text: str) -> str:
    en_punct = string.punctuation
    cn_punct = "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    return text.translate(str.maketrans("", "", en_punct + cn_punct))


def _split_text_to_tokens(text: str) -> list:
    text = _remove_punctuation(text)
    pattern = re.compile(r"[a-zA-Z]+|[^a-zA-Z]")
    return [m.group() for m in pattern.finditer(text) if m.group().strip()]


def _is_last_token_chinese(tokens: list) -> bool:
    if not tokens:
        return False
    return any("\u4e00" <= ch <= "\u9fff" for ch in tokens[-1])


def _run_streaming(
    asr,
    wav16k: np.ndarray,
    step_ms: int,
    chunk_size_sec: float,
    unfixed_token_num: int,
    lookahead_ms: int,
    language=None,
    context: str = "",
) -> tuple:
    """Streaming decode of a full utterance; returns (final_text, final_language)."""
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))
    lookahead = int(round(lookahead_ms / 1000.0 * sr))

    state = asr.init_streaming_state(
        context=context,
        language=language,
        unfixed_chunk_num=0,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
    )

    pos = 0
    new_text = ""
    last_text_tmp = ""
    total_new_asr_tokens = []

    max_new_tokens = max(1, int((step + lookahead) / 1280))
    first_max_new_tokens = max_new_tokens
    is_first = True

    max_new_tokens_floor = min(32, max(4, 2 * int(step / 1280)))
    while pos < wav16k.shape[0]:
        if is_first:
            seg = wav16k[pos : pos + step + lookahead]
            is_first = False
            state.chunk_size_sec = (step + lookahead) / sr
            state.chunk_size_samples = max(1, int(round(float(state.chunk_size_sec) * sr)))
        else:
            seg = wav16k[pos : pos + step]
            state.chunk_size_sec = chunk_size_sec
            state.chunk_size_samples = max(1, int(round(float(chunk_size_sec) * sr)))
        pos += seg.shape[0]

        _, text = asr.streaming_transcribe(seg, state, int(max_new_tokens))
        text = text.split("|")[0]

        if len(text) > len(last_text_tmp):
            last_text_tmp = text
            max_new_tokens = max(1, int(step / 1280))
        else:
            if not _is_last_token_chinese(total_new_asr_tokens):
                max_new_tokens = max_new_tokens + 1
            else:
                max_new_tokens = max(1, int(step / 1280))

        if _is_last_token_chinese(total_new_asr_tokens):
            max_new_tokens = 2 * max_new_tokens

        max_new_tokens = min(max_new_tokens_floor, max_new_tokens)

        if seg.shape[0] / sr < chunk_size_sec:
            break

    asr.finish_streaming_transcribe(state, first_max_new_tokens)
    return state.text.split("|")[0], state.language


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------
class R2T2ModelLoader:
    """Load the Confucius4-R2T2 ASR model (transformers or vLLM backend)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {"default": "netease-youdao/Confucius4-R2T2"},
                ),
                "backend": (["transformers", "vllm"], {"default": "transformers"}),
                "max_new_tokens": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 16}),
            },
            "optional": {
                "auto_download": ("BOOLEAN", {"default": True}),
                "device": ("STRING", {"default": "cuda:0"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "gpu_memory_utilization": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("R2T2_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Confucius4-R2T2/ASR"

    def load_model(
        self,
        model_path,
        backend,
        max_new_tokens,
        auto_download=True,
        device="cuda:0",
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
    ):
        local_path = _ensure_model(model_path, auto_download=bool(auto_download))
        key = (local_path, backend, max_new_tokens, device, dtype)
        if key in _MODEL_CACHE:
            return (_MODEL_CACHE[key],)

        R2T2ASRModel = _get_model_class()

        if backend == "vllm":
            try:
                asr = R2T2ASRModel.LLM(
                    model=local_path,
                    gpu_memory_utilization=gpu_memory_utilization,
                    max_new_tokens=int(max_new_tokens),
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load R2T2 with the vLLM backend. "
                    "Make sure `qwen-asr[vllm]` is installed and vLLM supports your OS/CUDA "
                    "(vLLM officially supports Linux only). "
                    "You can also switch the backend to 'transformers'.\n"
                    f"Original error: {e}"
                ) from e
        else:
            try:
                import torch  # noqa: PLC0415
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                asr = R2T2ASRModel.from_pretrained(
                    local_path,
                    dtype=dtype_map.get(dtype, torch.bfloat16),
                    device_map=device,
                    max_new_tokens=int(max_new_tokens),
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load R2T2 with the transformers backend. "
                    "Make sure `qwen-asr` is installed (pip install qwen-asr).\n"
                    f"Original error: {e}"
                ) from e

        _MODEL_CACHE[key] = asr
        return (asr,)


class R2T2Transcribe:
    """One-shot (non-streaming) transcription of a full audio clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("R2T2_MODEL",),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "language": ("STRING", {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "language")
    FUNCTION = "transcribe"
    CATEGORY = "Confucius4-R2T2/ASR"

    def transcribe(self, model, audio=None, audio_path="", language="auto", context=""):
        wav16k = _audio_input_to_16k(audio, audio_path)
        lang = _language_param(language)

        results = model.transcribe(
            audio=[(wav16k, 16000)],
            language=[lang],
            context=context or "",
            return_time_stamps=False,
        )
        r = results[0]
        return (str(r.text), str(r.language))


class R2T2StreamingTranscribe:
    """Streaming transcription of an utterance (vLLM backend only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("R2T2_MODEL",),
            },
            "optional": {
                "audio": ("AUDIO",),
                "audio_path": ("STRING", {"default": ""}),
                "language": ("STRING", {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "chunk_size_ms": ("INT", {"default": 160, "min": 40, "max": 2000, "step": 10}),
                "lookahead_ms": ("INT", {"default": 160, "min": 0, "max": 2000, "step": 10}),
                "unfixed_token_num": ("INT", {"default": 1, "min": 0, "max": 20, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "language")
    FUNCTION = "transcribe_streaming"
    CATEGORY = "Confucius4-R2T2/ASR"

    def transcribe_streaming(
        self,
        model,
        audio=None,
        audio_path="",
        language="auto",
        context="",
        chunk_size_ms=160,
        lookahead_ms=160,
        unfixed_token_num=1,
    ):
        if getattr(model, "backend", None) != "vllm":
            raise ValueError(
                "Streaming transcription requires the vLLM backend. "
                "Load the model with backend='vllm' (Linux + CUDA + qwen-asr[vllm] required)."
            )

        wav16k = _audio_input_to_16k(audio, audio_path)
        lang = _language_param(language)
        chunk_size_sec = int(chunk_size_ms) / 1000.0

        text, detected_lang = _run_streaming(
            model,
            wav16k,
            step_ms=int(chunk_size_ms),
            chunk_size_sec=chunk_size_sec,
            unfixed_token_num=int(unfixed_token_num),
            lookahead_ms=int(lookahead_ms),
            language=lang,
            context=context or "",
        )
        return (str(text), str(detected_lang))


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "R2T2ModelLoader": R2T2ModelLoader,
    "R2T2Transcribe": R2T2Transcribe,
    "R2T2StreamingTranscribe": R2T2StreamingTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "R2T2ModelLoader": "Confucius4-R2T2 Model Loader",
    "R2T2Transcribe": "Confucius4-R2T2 Transcribe (One-shot)",
    "R2T2StreamingTranscribe": "Confucius4-R2T2 Transcribe (Streaming)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
