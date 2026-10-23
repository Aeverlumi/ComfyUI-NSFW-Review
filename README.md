# ComfyUI-NSFW-Review

ComfyUI nodes for parsing model-produced content ratings, preparing prompt audits, and combining video and prompt assessments. Optional image-model nodes inspect still images or decoded video frames. The distributed package contains no model weights.

## Nodes

- `NSFWJSONNormalize`, `CivitaiJSONNormalize`: parse a JSON rating from model output.
- `BuildNSFWPromptAudit`, `BuildNSFWPromptAuditV2`, `BuildCivitaiPromptAudit`: prepare text-only audit instructions.
- `CombineNSFWAssessments`, `CombineCivitaiAssessments`: compare video and prompt ratings.
- `MetaSexualRiskScore`: map a video rating to a fixed reference score.
- `MetaApprovalEstimate`: map a video rating to editable, uncalibrated probability priors.
- `BuildMetaVideoAuditV0`, `BuildMetaTextAuditV0`: build prompts from `meta_sexual_standard_v0.json`.
- `MetaEvidenceNormalizeV0`, `MetaEvidenceReportV0`: validate model claims and produce a reviewable evidence draft without approval probabilities.
- `SexualApprovalRiskV1`, `AgeSafetyNormalize`: combine Qwen visual/prompt ratings with sampled Blue-Eye frame signals and flag age cues for human verification.

## Install

Place this folder at `ComfyUI/custom_nodes/ComfyUI-NSFW-Review` and restart ComfyUI. The original rating and evidence nodes use only Python's standard library. Connect a separate vision-language or text model to produce their JSON inputs. Image-model nodes require optional dependencies and weights described below.

## Scope

The scores and probability priors are illustrative mappings, not measured platform approval rates. Results depend on the upstream model's rating and any sampled frames. They do not account for audio, ad copy, targeting, landing pages, or account status. Review flagged or uncertain cases manually.

`SexualApprovalRiskV1` uses rating anchors `PG=0, PG13=45, R=70, X=90, XXX=100`. Its Blue-Eye frame component is `100 * (0.35 * suggestive + explicit)`; for video it combines `0.70 * peak_frame + 0.30 * p90_frame`. The final 0-100 sexual-content index is `round_half_up(0.55 * Qwen_visual + 0.35 * Blue-Eye + 0.10 * Qwen_prompt)`. It is not an overall ad approval probability or a policy compliance decision. Inspect flagged frames and verify age cues independently.

The v0 evidence workflow keeps old nodes available for prior workflows. Model findings in its report are unverified pointers; confirm them against the source video before using them as case labels. Keep `meta_sexual_standard_v0.json` alongside `__init__.py` when copying the plugin.

## Optional image model nodes

This same plugin also provides `OpenNSFW2ImageSignals`, `FalconsaiImageSignals`, and
`BlueEyeImageSignals`. They take a ComfyUI `IMAGE` batch (a still image or frames from
VHS), uniformly select up to `max_frames`, and return a JSON string plus the largest
raw model signal. OpenNSFW2 and Falconsai report `nsfw`; Blue-Eye reports
`suggestive + explicit`. These are **model scores, not Meta approval probabilities**.
No image model can derive an actual ad-review outcome from frames alone.
`CompareImageSignals` accepts any subset of the three JSON outputs, checks whether
their sampling aligns, and lists frames worth reviewing without voting on a rating.
`FormatImageSignalsReport` turns either a single-model JSON or the comparison JSON
into a Chinese plain-text report with each sampled frame's native predicted class,
class probabilities, class counts, and timestamps. The example workflows save this
readable report alongside the original JSON for automation. Blue-Eye's native result
is the highest of `safe`, `suggestive`, and `explicit`; Falconsai uses
`normal`/`nsfw`, and OpenNSFW2 uses `sfw`/`nsfw`. The JSON's `signal` and
`max_signal` fields remain auxiliary frame-selection values for compatibility;
they are not author-provided video ratings. In particular, the Blue-Eye auxiliary
signal sums `suggestive + explicit`, while its native prediction is the largest
single class probability. No model class is mapped to PG/PG13/R/X/XXX automatically.

The JSON records every sampled input-frame index, class probabilities, and an
approximate timestamp. Set `fps` to the effective frame rate of the incoming batch
(for the current VHS graph, 14). Qwen's own `max_frames` does not limit these nodes.
If VHS skips frames or loads a segment, the timestamps are relative estimates; use
the frame indices and original video to confirm any finding. The model nodes do not
replace the existing five-level Qwen assessment.

### ShieldGemma 2 policy review

`ShieldGemma2PolicySignals` is an independent image-policy node. Connect a still
image or a VHS `IMAGE` frame batch. The node samples up to `max_frames` and outputs
per-frame `Yes`/`No` probabilities, the native highest-probability label, an editable
threshold flag, JSON, a Chinese report, and the selected policy text. Choose among
Google's three built-in policies (`official_sexual`, `official_dangerous`,
`official_violence`), three **project draft** policies (`facebook_ads_visual`,
`azure_aws_visual`, and the more detailed `detailed_sexual_visual`), and `custom`.
`detailed_sexual_visual` uses the project's visual evidence and counterexamples
without asking the model to assign a PG/PG13/R/X/XXX rating. The read-only preview shows the full selected
policy before execution; `custom_policy` is used only in `custom` mode. The exact
policy text and its source are also recorded in the JSON and report. The draft
policies borrow visual distinctions from Meta's ad rules and Azure/AWS moderation
taxonomies; they are not official Meta, Azure, or AWS policies and have not been
validated as more accurate. The default `official_sexual` policy targets **explicit
sexual content**, so it is not a comprehensive detector of suggestive advertising.
For separate questions, run separate nodes; custom wording changes model behavior
and needs validation on your examples.
The `review_threshold` only flags frames for review. It does not change the
model's native `Yes`/`No` prediction. `Yes` probability is the first item of
ShieldGemma 2's official `probabilities` output. Neither probability is a measured
Meta approval chance.

The node loads the model only when it runs and releases it afterward. It never
downloads weights during inference. Install the optional torch requirements in
ComfyUI's Python environment, then accept the license for
[google/shieldgemma-2-4b-it](https://huggingface.co/google/shieldgemma-2-4b-it)
with your own Hugging Face account and download the complete repository into
`ComfyUI/models/meta_nsfw_review/shieldgemma2`. For example, after authenticating:

```bash
hf download google/shieldgemma-2-4b-it --local-dir ComfyUI/models/meta_nsfw_review/shieldgemma2
```

Alternatively, set the node's `model_path` to an existing complete local copy.
Leave it blank to use the standard path (or `META_NSFW_MODEL_DIR/shieldgemma2`).
The original BF16 weights alone are about 8.6 GB; reserve additional disk for
tokenizer files and download overhead, and enough free GPU memory for inference.
Restart ComfyUI after installing the node when active work has finished. Existing
workflows do not use this node unless you explicitly add it.

After activation, import `examples/shieldgemma2_video_gui.json` for a separate
video review graph; `examples/shieldgemma2_video_api.json` is its API equivalent.
Both save the readable report and raw JSON under `ComfyUI/output/video_review`.

### SafeWatch-8B video review

`SafeWatch8BVideoReview` accepts a VHS `IMAGE` batch and reviews up to 48 frames.
It keeps half the samples spread across the video and fills the rest with frames
at the largest visual changes. It sends overlapping eight-frame segments to the
model in one worker process, retaining their order. It returns per-segment answers,
a structured JSON report and a readable report. `original_categories` uses the
SafeWatch authors' unmodified `video_moderation_prompt_vg_fine_grained_final`
six-category prompt and is the default. It is not Meta's advertising policy.
`sexual_focus` is a project-specific sexual-content prompt; `custom` accepts your
own text. The JSON `sexual_content_flag` is nullable when the answer cannot be
parsed. It is a video-level model judgment, not a per-frame probability or a Meta
approval estimate. Audio and advertising context are not included.
The standalone example uses VHS to decode at 14 fps and scale frames to 448
pixels high before making the ComfyUI image batch. It inspects up to 48 frames.
When the decoded batch exceeds 48 frames, the
hybrid sampler keeps timeline coverage while adding visually changing frames.
The category summary flags any positive segment and
retains uncertainty if a segment cannot be parsed. The JSON and readable report
include elapsed time and the worker's peak CUDA allocation; this excludes memory
held by ComfyUI or other processes.

Install the author's [SafeWatch-8B checkpoint](https://huggingface.co/Virtue-AI-HUB/SafeWatch-8B)
in `ComfyUI/models/meta_nsfw_review/safewatch8b` (about 16.2 GB), or enter an
existing local path in `model_path`. The node never downloads during execution.
Its checkpoint uses custom InternVL model code; review the repository code before
using `trust_remote_code` on a new version. Import
`examples/safewatch8b_video_gui.json` after ComfyUI has loaded the new node.
The published InternVL code requires an older Transformers runtime than this
instance's ComfyUI. Install it in a separate environment at
`ComfyUI/custom_nodes/ComfyUI-NSFW-Review/.venv-safewatch`:

```bash
python -m venv --system-site-packages ComfyUI/custom_nodes/ComfyUI-NSFW-Review/.venv-safewatch
ComfyUI/custom_nodes/ComfyUI-NSFW-Review/.venv-safewatch/bin/python -m pip install \
  'transformers==4.44.2' 'huggingface-hub<1' 'tokenizers<0.20' 'sentencepiece==0.2.0'
```

The node starts a one-shot worker from this environment; set `SAFEWATCH_PYTHON`
to a different interpreter path if needed. No packages are changed in ComfyUI's
main environment.

### Install dependencies and models

Copy this whole directory into `ComfyUI/custom_nodes/ComfyUI-NSFW-Review`. Use the
Python environment that launches ComfyUI. Install only the backends you plan to use:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-NSFW-Review/requirements-torch.txt
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-NSFW-Review/requirements-opennsfw2.txt
```

The second command is optional and installs TensorFlow. It may need a separate
environment on ComfyUI installations with conflicting dependencies. Install model
files with the included script:

```bash
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/install_models.py --model opennsfw2 --comfyui-root ComfyUI
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/install_models.py --model falconsai --comfyui-root ComfyUI
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/install_models.py --model blue_eye --comfyui-root ComfyUI
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/install_models.py --verify-only --comfyui-root ComfyUI
```

Blue-Eye is gated. Accept the license on its [model page](https://huggingface.co/pp1618/Blue-Eye)
and run `hf auth login` on the installation machine before downloading it. The script
never embeds a token. For a nonstandard model location use `--models-dir PATH`, or
set `META_NSFW_MODEL_DIR` for both installation and ComfyUI runtime. Directory names
are fixed: `opennsfw2`, `falconsai`, `blue_eye` beneath `meta_nsfw_review`. Each has
a `manifest.json` of file hashes. The installer does not download during inference;
missing or corrupted files produce an actionable error. Model files and credentials
are not part of the distributed plugin.

Run the three nodes one at a time at first and compare their native classes on the same input
frames. Their first invocation includes model loading time. On a shared GPU, keep
`batch_size` small, especially for Blue-Eye. OpenNSFW2 uses TensorFlow while the
other two use PyTorch; check peak GPU memory on the target host before increasing
batch size or leaving several models resident.

### API example

The `examples` directory has four importable GUI workflows: `opennsfw2_video_gui.json`,
`falconsai_video_gui.json`, `blue_eye_video_gui.json`, and `all_video_gui.json`.
Open one in ComfyUI, choose a file in its VHS video loader, then queue it. Each
independent workflow decodes the whole video at 14 fps and uniformly samples up to
48 frames. The combined workflow starts with eight frames for a quick check; raise
its VHS `frame_load_cap` to 0 and each model's `max_frames` to 48 for a full run.
Each GUI file has a matching `_api.json` file for `/prompt` automation. Outputs
are text files under `ComfyUI/output/video_review`: open files ending in `_report_*.txt`
for the readable result; files ending in `_signals_*.txt` or `model_comparison_*.txt`
contain the raw JSON. OpenNSFW2 defaults to CPU to
avoid TensorFlow and PyTorch competing for GPU memory. `force_rate` and
`frame_load_cap` control VHS decoding; each model node's `max_frames` controls
uniform selection within that decoded batch. To test a still image, replace the
VHS video loader with ComfyUI's `LoadImage` node and connect its `IMAGE` output.

For a ComfyUI listening on its default Vast.ai internal port, submit and wait with:

```bash
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/examples/run_video_signals.py example.mp4 --model opennsfw2
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/examples/run_video_signals.py example.mp4 --model falconsai
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/examples/run_video_signals.py example.mp4 --model blue_eye
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/examples/run_video_signals.py example.mp4
python ComfyUI/custom_nodes/ComfyUI-NSFW-Review/examples/run_video_signals.py example.mp4 --full-video
```

The individual commands decode the whole video and sample up to 48 frames per model.
Use `--base-url` for a different local ComfyUI URL. The script prints each model's
Chinese report and saved output filenames, or the ComfyUI error if execution fails.
