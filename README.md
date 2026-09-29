# ComfyUI-Confucius4-R2T2

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) 自定义节点包，将网易有道开源的
**Confucius4-R2T2**（多语种流式 / 一次性 ASR 语音识别模型，基于 Qwen3-ASR 架构）接入 ComfyUI 工作流。

支持中文、英文、日语、韩语、法语、德语、意大利语、葡萄牙语、俄语、西班牙语、阿拉伯语等多语种转写，
以及 **80ms ~ 2s 分片流式实时转写**（vLLM 后端）。

> 上游项目：https://github.com/netease-youdao/Confucius4-R2T2 （Apache-2.0）

---

## 功能

| 节点 | 类型 | 说明 |
| --- | --- | --- |
| `Confucius4-R2T2 Model Loader` | 模型加载 | 加载模型，支持 `transformers` / `vllm` 双后端 |
| `Confucius4-R2T2 Transcribe (One-shot)` | 一次性转写 | 整段音频一次性输出文本（两种后端均支持） |
| `Confucius4-R2T2 Transcribe (Streaming)` | 流式转写 | 按分片流式输出文本（仅 vLLM 后端） |

### 输入说明

- **audio**（`AUDIO` 类型）：可接入 VideoHelperSuite 等社区的音频输出节点；
- **audio_path**（`STRING` 类型）：直接填写本地音频文件路径（wav / mp3 / flac 等，librosa 会自动解码并重采样到 16 kHz）；
- 两者至少提供其一，同时提供时优先使用 `audio` 输入。

### 参数说明

- **model_path**：Hugging Face 仓库 ID（默认 `netease-youdao/Confucius4-R2T2`）或本地模型目录路径；
- **auto_download**：`True` 时若 `model_path` 为 HF 仓库 ID 且本地无此模型，会自动从 Hugging Face 下载到 `ComfyUI/models/r2t2/` 下（默认开启）；
- **backend**：`transformers`（默认，Windows / CPU 可用）或 `vllm`（Linux + CUDA 加速，流式必需）；
- **language**：`auto` 自动检测，或指定 `Chinese` / `English` / `Japanese` / `Korean` / `French` / `German` / `Italian` / `Portuguese` / `Russian` / `Spanish` / `Arabic` 等；
- **context**：热词 / 上下文提示；
- **chunk_size_ms / lookahead_ms / unfixed_token_num**：流式分片参数（默认 160 / 160 / 1，与上游示例一致）。

---

## 模型存储与下载

**模型权重不包含在本插件仓库中**（仓库本体约 66 KB，远小于 50 MB，可通过 ComfyUI-Manager 链接直接安装）。
模型统一存放在 ComfyUI 的 **`ComfyUI/models/r2t2/`** 目录下，目录结构：

```
ComfyUI/models/r2t2/
└── netease-youdao/
    └── Confucius4-R2T2/     ← 模型权重（config.json + safetensors 等）
```

### 自动下载（默认）

`model_path` 填 Hugging Face 仓库 ID（如 `netease-youdao/Confucius4-R2T2`），保持 `auto_download=True`，
节点首次执行时会自动下载权重到上述目录；下载完成后自动从本地路径加载。已下载过则直接复用，不会重复下载。

自动下载自带 **hf-mirror.com 镜像回退**：默认直连 Hugging Face 失败时，自动改用
`https://hf-mirror.com`（国内无需代理），无需任何配置。两者都失败时会给出手动下载命令。

### 手动下载（可选）

也可以在任何时间手动下载，效果相同：

```bash
# 方式一：huggingface-cli + 国内镜像（推荐，无需代理）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download netease-youdao/Confucius4-R2T2 \
  --local-dir "ComfyUI/models/r2t2/netease-youdao/Confucius4-R2T2"

# 方式二：huggingface-cli 直连（海外网络 / 已配置代理）
huggingface-cli download netease-youdao/Confucius4-R2T2 \
  --local-dir "ComfyUI/models/r2t2/netease-youdao/Confucius4-R2T2"

# 方式三：git lfs
git lfs install
git clone https://huggingface.co/netease-youdao/Confucius4-R2T2 \
  "ComfyUI/models/r2t2/netease-youdao/Confucius4-R2T2"
```

手动下载后，`model_path` 继续填仓库 ID（自动识别本地已有），或直接填本地目录路径均可。
若关闭 `auto_download` 且本地不存在，节点会给出上述手动下载命令提示。

---

## 安装

### 方式一：ComfyUI-Manager（推荐）

在 ComfyUI-Manager 中通过 Git URL 安装：

```
https://github.com/dsAI-Q/ComfyUI-Confucius4-R2T2-dashengAi.git
```

### 方式二：手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dsAI-Q/ComfyUI-Confucius4-R2T2-dashengAi.git
cd ComfyUI-Confucius4-R2T2
pip install -r requirements.txt
```

重启 ComfyUI 后，节点会出现在 **`Confucius4-R2T2/ASR`** 分类下。

> **vLLM 加速（可选，仅 Linux + CUDA）**：
> ```bash
> pip install -U qwen-asr[vllm]
> ```
> 加载模型时选择 `backend="vllm"` 即可启用流式转写。

---

## 工作流示例

```
[Load Audio (VHS)] ──► [R2T2 Model Loader] ──► [R2T2 Transcribe (One-shot)] ──► [Show Text]
                          backend=transformers        audio = AUDIO
```

- 一次性转写：`Model Loader (backend=transformers)` → `Transcribe`，输入 `audio` 或 `audio_path`，输出 `text` / `language`。
- 流式转写：`Model Loader (backend=vllm)` → `Streaming Transcribe`（仅支持 vLLM 后端）。

---

## 注意事项

1. **模型下载**：默认自动从 Hugging Face 下载到 `ComfyUI/models/r2t2/`（约 2B 参数，需可访问 HF）；也可手动下载，详见上方「模型存储与下载」章节。
2. **vLLM 仅支持 Linux**：Windows / macOS 请使用 `transformers` 后端（一次性转写不受影响）。
3. **流式转写限制**：仅 vLLM 后端；不支持时间戳；单条音频（无批处理）。
4. **显存**：transformers 后端约需 8GB+ 显存（bfloat16），vLLM 后端可通过 `gpu_memory_utilization` 调节。

---

## 许可与致谢

- 插件代码与打包结构：Apache-2.0（本仓库 `LICENSE`）。
- `r2t2/` 目录为上游 [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) 的薄封装源码（Apache-2.0），模型推理依赖 [Qwen-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 `qwen-asr` 包。
- 模型权重许可见上游仓库 `MODEL_LICENSE`。
