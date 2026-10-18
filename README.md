# ComfyUI-NSFW-Review

ComfyUI utility nodes for parsing model-produced content ratings, preparing prompt audits, and combining video and prompt assessments. The package contains no model weights and does not inspect media by itself.

## Nodes

- `NSFWJSONNormalize`, `CivitaiJSONNormalize`: parse a JSON rating from model output.
- `BuildNSFWPromptAudit`, `BuildNSFWPromptAuditV2`, `BuildCivitaiPromptAudit`: prepare text-only audit instructions.
- `CombineNSFWAssessments`, `CombineCivitaiAssessments`: compare video and prompt ratings.
- `MetaSexualRiskScore`: map a video rating to a fixed reference score.
- `MetaApprovalEstimate`: map a video rating to editable, uncalibrated probability priors.

## Install

Place this folder at `ComfyUI/custom_nodes/ComfyUI-NSFW-Review` and restart ComfyUI. The nodes use only Python's standard library. Connect a separate vision-language or text model to produce the JSON inputs.

## Scope

The scores and probability priors are illustrative mappings, not measured platform approval rates. Results depend on the upstream model's rating and any sampled frames. They do not account for audio, ad copy, targeting, landing pages, or account status. Review flagged or uncertain cases manually.
