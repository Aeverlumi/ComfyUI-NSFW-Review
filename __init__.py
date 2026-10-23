import json
import sys
from pathlib import Path

_plugin_dir = str(Path(__file__).resolve().parent)
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

from image_models import NODE_CLASS_MAPPINGS as IMAGE_NODE_CLASS_MAPPINGS
from image_models import NODE_DISPLAY_NAME_MAPPINGS as IMAGE_NODE_DISPLAY_NAME_MAPPINGS
from shieldgemma2_nodes import NODE_CLASS_MAPPINGS as SHIELDGEMMA_NODE_CLASS_MAPPINGS
from shieldgemma2_nodes import NODE_DISPLAY_NAME_MAPPINGS as SHIELDGEMMA_NODE_DISPLAY_NAME_MAPPINGS
from safewatch8b_nodes import NODE_CLASS_MAPPINGS as SAFEWATCH_NODE_CLASS_MAPPINGS
from safewatch8b_nodes import NODE_DISPLAY_NAME_MAPPINGS as SAFEWATCH_NODE_DISPLAY_NAME_MAPPINGS
from approval_risk import NODE_CLASS_MAPPINGS as APPROVAL_RISK_NODE_CLASS_MAPPINGS
from approval_risk import NODE_DISPLAY_NAME_MAPPINGS as APPROVAL_RISK_NODE_DISPLAY_NAME_MAPPINGS
from age_safety import NODE_CLASS_MAPPINGS as AGE_SAFETY_NODE_CLASS_MAPPINGS
from age_safety import NODE_DISPLAY_NAME_MAPPINGS as AGE_SAFETY_NODE_DISPLAY_NAME_MAPPINGS


class NSFWJSONNormalize:
    CATEGORIES = {"SAFE", "SUGGESTIVE", "NUDITY", "SEXUAL_ACTIVITY", "UNCERTAIN"}
    RATINGS = {"GENERAL", "SENSITIVE", "QUESTIONABLE", "EXPLICIT", "UNCERTAIN"}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw_text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "category")
    FUNCTION = "normalize"
    CATEGORY = "video/review"

    def normalize(self, raw_text):
        result = None
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw_text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break

        rating = result.get("rating") if isinstance(result, dict) else None
        category = result.get("category") if isinstance(result, dict) else None
        if rating not in self.RATINGS and category not in self.CATEGORIES:
            result = {
                "rating": "UNCERTAIN",
                "review_required": True,
                "evidence": "模型未返回可解析的分类结果",
                "limitations": "需要人工复核原视频",
            }

        rating = result.get("rating")
        category = result.get("category")
        label = rating if rating in self.RATINGS else category
        if rating in self.RATINGS:
            is_nsfw = None if rating == "UNCERTAIN" else rating in {"QUESTIONABLE", "EXPLICIT"}
        else:
            is_nsfw = None if category == "UNCERTAIN" else category != "SAFE"
        evidence = result.get("evidence") or ""
        if isinstance(evidence, list):
            evidence = "；".join(str(item) for item in evidence)
        normalized = {
            "is_nsfw": is_nsfw,
            "rating": rating if rating in self.RATINGS else None,
            "category": category if category in self.CATEGORIES else label,
            "review_required": label == "UNCERTAIN" or result.get("review_required") is True,
            "evidence": str(evidence)[:500],
            "limitations": str(result.get("limitations") or "")[:500],
        }
        return (json.dumps(normalized, ensure_ascii=False), label)


class BuildNSFWPromptAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse_prompt": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audit_prompt",)
    FUNCTION = "build"
    CATEGORY = "video/review"

    def build(self, reverse_prompt):
        instruction = (
            "只审核下方文生视频提示词的字面内容，不推断原视频。"
            "按性内容分级：GENERAL=没有性化描述；SENSITIVE=明确性感、挑逗或轻度性暗示，"
            "但没有裸露或直接性接触；QUESTIONABLE=明确描写裸露乳房或臀部、近裸体、"
            "性化身体接触或模拟性行为；EXPLICIT=明确描写生殖器或性行为；"
            "UNCERTAIN=文本无法判断。"
            "职业装、女仆装、丝袜、短裙、腮红、害羞或惊讶表情、换装情节，"
            "以及有人向人物伸手，本身都不是 NSFW 证据；"
            "没有明确性行为、裸露或直接性化动作时，应判 GENERAL。"
            "提到大腿、张腿或身体轮廓本身不等于裸体。"
            "多类并存时选明确写出的最严重类别。"
            "若提示词出现年龄不明或可能未成年人的性化内容，review_required=true。"
            "只输出一个 JSON 对象，字段为 rating、review_required、evidence、limitations。"
            "不要输出 is_nsfw；后续节点按 QUESTIONABLE 阈值计算。"
            "evidence 简短引用提示词含义，不要编造原视频内容。"
            "limitations 说明本次仅审核提示词，不能证明原视频安全。\n\n"
            "待审核提示词：\n"
        )
        return (instruction + reverse_prompt,)


class BuildNSFWPromptAuditV2(BuildNSFWPromptAudit):
    def build(self, reverse_prompt):
        instruction = (
            "只审核下方文生视频提示词明确写出的内容，不推断原视频。"
            "按性内容分级：GENERAL=没有明确性化描述的普通场景；"
            "SENSITIVE=明确的轻度性感或性暗示，包括刻意性感摆姿、性感部位特写、"
            "内衣或明显暴露的服装，但没有明确裸露或直接性化身体接触；"
            "QUESTIONABLE=明确描写裸露乳房或臀部、近裸体、性化身体接触、"
            "模拟性行为或明显软色情；EXPLICIT=明确描写生殖器或性行为；"
            "UNCERTAIN=文本不足以可靠评级。"
            "职业装、女仆装、丝袜、短裙、腮红、害羞或惊讶表情、换装情节、"
            "以及有人向人物伸手，本身不足以升级评级；"
            "大腿、张腿或身体轮廓本身不是裸露证据。"
            "不要仅凭服装名称推断裸体或性接触，也不要把明确的轻度性感压成 GENERAL。"
            "多类并存时选有文字证据的最高级别；证据不清时选 UNCERTAIN，"
            "不要为了给出确定结论而猜测。"
            "若提示词出现年龄不明或可能未成年人的性化内容，review_required=true。"
            "只输出一个 JSON 对象，字段为 rating、review_required、evidence、limitations。"
            "不要输出 is_nsfw；后续节点按 QUESTIONABLE 阈值计算。"
            "evidence 简短引用提示词含义，不要编造原视频内容。"
            "limitations 说明本次仅审核提示词，不能证明原视频安全。\n\n"
            "待审核提示词：\n"
        )
        return (instruction + reverse_prompt,)


class CombineNSFWAssessments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_json": ("STRING", {"forceInput": True}),
                "prompt_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_json",)
    FUNCTION = "combine"
    CATEGORY = "video/review"

    def combine(self, video_json, prompt_json):
        try:
            video = json.loads(video_json)
            prompt = json.loads(prompt_json)
            video_result = video["is_nsfw"]
            prompt_result = prompt["is_nsfw"]
            if video_result not in (True, False, None) or prompt_result not in (True, False, None):
                raise ValueError("invalid verdict")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            video = {"category": "UNCERTAIN", "rating": "UNCERTAIN", "evidence": ""}
            prompt = {"category": "UNCERTAIN", "rating": "UNCERTAIN", "evidence": ""}
            video_result = None
            prompt_result = None

        disagreement = video_result is not None and prompt_result is not None and video_result != prompt_result
        result = {
            "is_nsfw": video_result,
            "video_rating": video.get("rating") or video.get("category"),
            "prompt_rating": prompt.get("rating") or prompt.get("category"),
            "nsfw_threshold": "QUESTIONABLE" if video.get("rating") else "legacy_category",
            "independent_visual_confirmation": False,
            "prompt_omits_video_nsfw": video_result is True and prompt_result is False,
            "prompt_adds_unseen_nsfw": video_result is False and prompt_result is True,
            "review_required": (
                video_result is None
                or prompt_result is None
                or disagreement
                or video.get("review_required") is True
                or prompt.get("review_required") is True
            ),
            "video_evidence": video.get("evidence", ""),
            "prompt_evidence": prompt.get("evidence", ""),
            "limitations": "视频判定仅覆盖抽样帧；提示词判定只检查反推文字，不能推翻视频判定。",
        }
        return (json.dumps(result, ensure_ascii=False),)


class CivitaiJSONNormalize:
    RATINGS = {"PG", "PG13", "R", "X", "XXX", "UNCERTAIN"}

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw_text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "rating")
    FUNCTION = "normalize"
    CATEGORY = "video/review"

    def normalize(self, raw_text):
        result = None
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw_text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(raw_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                result = candidate
                break

        if isinstance(result, dict) and isinstance(result.get("rating"), str):
            result["rating"] = result["rating"].upper().replace("-", "").replace(" ", "")

        if not isinstance(result, dict) or result.get("rating") not in self.RATINGS:
            result = {
                "rating": "UNCERTAIN",
                "review_required": True,
                "evidence": "模型未返回可解析的 PG/PG13/R/X/XXX 分类结果",
                "limitations": "需要人工复核原始内容",
            }

        rating = result["rating"]
        evidence = result.get("evidence") or ""
        if isinstance(evidence, list):
            evidence = "；".join(str(item) for item in evidence)
        normalized = {
            "is_nsfw": None if rating == "UNCERTAIN" else rating in {"R", "X", "XXX"},
            "rating": rating,
            "review_required": rating == "UNCERTAIN" or result.get("review_required") is True,
            "evidence": str(evidence)[:500],
            "limitations": str(result.get("limitations") or "")[:500],
        }
        return (json.dumps(normalized, ensure_ascii=False), rating)


class BuildCivitaiPromptAudit(BuildNSFWPromptAudit):
    def build(self, reverse_prompt):
        instruction = (
            "只审核下方文生视频提示词明确写出的性内容，不推断原视频，也不评级暴力、吸烟等其他内容。"
            "采用 Civitai 风格的五级标签：PG=普通或轻微暗示，没有明确性化呈现；"
            "PG13=明确性感或挑逗、内衣或暴露服装、局部裸露，但未明确写出裸露乳房、臀部、"
            "生殖器或性行为；R=明确写出裸露乳房、臀部或生殖器，但没有性行为；"
            "X=明确写出性行为或无歧义的性化身体接触，但没有细节特写；"
            "XXX=明确写出图像化、细节化的性行为。"
            "服装名称、深领口、身体轮廓、递东西、靠近、普通触碰、吸烟或暧昧氛围，"
            "不能单独证明裸体或性行为。若只写出这些内容，最高为 PG13。"
            "多类并存时选有文字证据的最高级别；高一级所需的证据不明确时，保持已确认的较低级别，"
            "并设 review_required=true。文本完全不足以判断时选 UNCERTAIN。"
            "若文字涉及可能未成年人且出现性化内容，设 review_required=true，不推测年龄或细节。"
            "只输出 JSON 对象，字段为 rating、review_required、evidence、limitations；"
            "rating 只能是 PG、PG13、R、X、XXX、UNCERTAIN。"
            "evidence 简短引用提示词的实际含义；limitations 说明仅审核文字，不能证明视频安全。"
            "不要输出 is_nsfw。\n\n待审核提示词：\n"
        )
        return (instruction + reverse_prompt,)


class BuildCivitaiImagePromptAudit(BuildCivitaiPromptAudit):
    CATEGORY = "image/review"

    def build(self, reverse_prompt):
        instruction = (
            "只审核下方文生图提示词明确写出的性内容，不推断原图，也不评级暴力等其他内容。"
            "采用 Civitai 风格的五级标签：PG=普通或轻微暗示，没有明确性化呈现；"
            "PG13=明确性感或挑逗、内衣或暴露服装、局部裸露，但未明确写出裸露乳房、臀部、"
            "生殖器或性行为；R=明确写出裸露乳房、臀部或生殖器，但没有性行为；"
            "X=明确写出性行为或无歧义的性化身体接触，但没有细节特写；"
            "XXX=明确写出图像化、细节化的性行为。"
            "仅凭服装名称、身体轮廓、普通触碰或暧昧氛围不能推断裸体或性行为。"
            "证据不足时保持已确认的较低级别并设 review_required=true；完全无法判断时选 UNCERTAIN。"
            "若文字涉及可能未成年人且出现性化内容，设 review_required=true，不推测年龄。"
            "只输出 JSON 对象，字段为 rating、review_required、evidence、limitations；"
            "rating 只能是 PG、PG13、R、X、XXX、UNCERTAIN。"
            "limitations 说明仅审核文字，不能证明原图安全。不要输出 is_nsfw。\n\n待审核提示词：\n"
        )
        return (instruction + reverse_prompt,)


class CombineCivitaiAssessments(CombineNSFWAssessments):
    def combine(self, video_json, prompt_json):
        result = json.loads(super().combine(video_json, prompt_json)[0])
        result["nsfw_threshold"] = "R"
        return (json.dumps(result, ensure_ascii=False),)


class MetaSexualRiskScore:
    BASE_SCORES = {"PG": 97, "PG13": 88, "R": 35, "X": 10, "XXX": 0}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_json": ("STRING", {"forceInput": True}),
                "prompt_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "score"
    CATEGORY = "video/review"

    def score(self, video_json, prompt_json):
        try:
            video = json.loads(video_json)
            prompt = json.loads(prompt_json)
            if not isinstance(video, dict) or not isinstance(prompt, dict):
                raise ValueError("assessments must be JSON objects")
        except (TypeError, ValueError, json.JSONDecodeError):
            video = {}
            prompt = {}
        rating = video.get("rating")
        if rating not in self.BASE_SCORES:
            rating = "UNCERTAIN"
        prompt_rating = prompt.get("rating")
        if prompt_rating not in (*self.BASE_SCORES, "UNCERTAIN"):
            prompt_rating = "UNCERTAIN"

        score = self.BASE_SCORES.get(rating)
        rating_disagreement = rating != prompt_rating and prompt_rating != "UNCERTAIN"
        review_required = (
            rating == "UNCERTAIN"
            or prompt_rating == "UNCERTAIN"
            or video.get("review_required") is True
            or prompt.get("review_required") is True
            or rating_disagreement
        )
        report = {
            "score": score,
            "score_scope": "video_sexual_content_only",
            "score_version": "rating_anchor_v1",
            "calibrated_to_meta_approval": False,
            "video_rating": rating,
            "prompt_rating": prompt_rating,
            "rating_disagreement": rating_disagreement,
            "review_required": review_required,
            "scoring_basis": "fixed_mapping_from_video_rating",
            "video_evidence": video.get("evidence", ""),
            "prompt_evidence": prompt.get("evidence", ""),
            "limitations": "仅覆盖抽样静帧和反推文字；没有音频、广告文案、落地页或 Meta 实际审核结果。分数按视频评级固定映射，不是过审概率。",
        }
        return (json.dumps(report, ensure_ascii=False),)


class MetaApprovalEstimate:
    DEFAULT_PRIORS = {"PG": 90, "PG13": 60, "R": 5, "X": 1, "XXX": 0}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score_json": ("STRING", {"forceInput": True}),
                "pg_prior": ("INT", {"default": 90, "min": 0, "max": 100}),
                "pg13_prior": ("INT", {"default": 60, "min": 0, "max": 100}),
                "r_prior": ("INT", {"default": 5, "min": 0, "max": 100}),
                "x_prior": ("INT", {"default": 1, "min": 0, "max": 100}),
                "xxx_prior": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "estimate"
    CATEGORY = "video/review"

    def estimate(self, score_json, pg_prior=90, pg13_prior=60, r_prior=5, x_prior=1, xxx_prior=0):
        try:
            score = json.loads(score_json)
            if not isinstance(score, dict):
                score = {}
        except (TypeError, ValueError):
            score = {}

        priors = dict(zip(self.DEFAULT_PRIORS, (pg_prior, pg13_prior, r_prior, x_prior, xxx_prior)))
        rating = score.get("video_rating")
        if not isinstance(rating, str) or rating not in priors:
            rating = "UNCERTAIN"
        review_flag = score.get("review_required")
        percent = priors.get(rating) if type(review_flag) is bool else None
        review_required = review_flag is not False or percent is None
        report = {
            "estimated_approval_probability_percent": percent,
            "estimate_scope": "sexual_content_of_video_only",
            "estimate_version": "editable_rating_priors_v1",
            "calibrated_to_meta_outcomes": False,
            "video_rating": rating,
            "prompt_rating": score.get("prompt_rating"),
            "rating_priors_percent": priors,
            "review_required": review_required,
            "estimate_reliability": "low_review_required" if review_required else "uncalibrated_prior",
            "video_evidence": score.get("video_evidence", ""),
            "limitations": "仅依据现有视频性内容评级映射可调整的初始概率；未检查音频、广告文案、定向、落地页、账户状态，也没有 Meta 历史审核标签。百分比未经校准，不是实际通过率。",
        }
        return (json.dumps(report, ensure_ascii=False),)


class MetaImageSexualRiskScore(MetaSexualRiskScore):
    CATEGORY = "image/review"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_json": ("STRING", {"forceInput": True}),
            "prompt_json": ("STRING", {"forceInput": True}),
        }}

    def score(self, image_json, prompt_json):
        report = json.loads(super().score(image_json, prompt_json)[0])
        report["score_scope"] = "image_sexual_content_only"
        report["scoring_basis"] = "fixed_mapping_from_image_rating"
        report["image_rating"] = report.pop("video_rating")
        report["image_evidence"] = report.pop("video_evidence")
        report["limitations"] = (
            "仅覆盖单张图片和反推文字；没有广告文案、落地页或 Meta 实际审核结果。"
            "分数按图片评级固定映射，不是过审概率。"
        )
        return (json.dumps(report, ensure_ascii=False),)


class MetaImageApprovalEstimate(MetaApprovalEstimate):
    CATEGORY = "image/review"

    def estimate(self, score_json, pg_prior=90, pg13_prior=60, r_prior=5, x_prior=1, xxx_prior=0):
        try:
            score = json.loads(score_json)
            if not isinstance(score, dict):
                score = {}
        except (TypeError, ValueError):
            score = {}
        adapted = dict(score, video_rating=score.get("image_rating"), video_evidence=score.get("image_evidence", ""))
        report = json.loads(super().estimate(
            json.dumps(adapted, ensure_ascii=False), pg_prior, pg13_prior, r_prior, x_prior, xxx_prior
        )[0])
        report["estimate_scope"] = "sexual_content_of_image_only"
        report["image_rating"] = report.pop("video_rating")
        report["image_evidence"] = report.pop("video_evidence")
        report["limitations"] = (
            "仅依据单张图片性内容评级映射可调整的初始概率；未检查广告文案、定向、"
            "落地页、账户状态，也没有 Meta 历史审核标签。百分比未经校准，不是实际通过率。"
        )
        return (json.dumps(report, ensure_ascii=False),)


def _meta_standard_v0():
    path = Path(__file__).with_name("meta_sexual_standard_v0.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_rating_guide(standard):
    return "；".join(f"{item['id']}={item['meaning']}" for item in standard["rating_scale"])


def _meta_rule_guide(standard):
    return "；".join(
        f"{rule['id']} {rule['signal']}。反例：{'、'.join(rule['counterexamples'])}"
        for rule in standard["rules"]
    )


class BuildMetaVideoAuditV0:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audit_prompt",)
    FUNCTION = "build"
    CATEGORY = "video/review"

    def build(self):
        standard = _meta_standard_v0()
        prompt = (
            f"审核标准版本 {standard['version']}。只判断视频抽样静帧中可见的成人性内容，不判断 Meta 是否过审。"
            f"内部描述性等级：{_meta_rating_guide(standard)}。"
            f"规则：{_meta_rule_guide(standard)}。"
            "只有明确可见的证据才能升级。服装、轮廓、胸口特写、靠近或递物不能自动证明裸露或性接触。"
            "判 PG13 前必须指出独立于服装名称、丝袜、领口、腮红和普通互动的明确性化镜头或动作；"
            "没有这种独立证据时，不得把上述元素写成 VIS-01，应判 PG 并用 VIS-06 记录可见内容。"
            "边界例：人物换上女仆装、穿丝袜、双臂交叉，另一人的手靠近或递东西，"
            "若没有性感部位持续特写、明确挑逗动作或性化接触，评级是 PG。"
            "反例：镜头持续以胸部或臀部为主体并明显强调该部位，可按 VIS-01 考虑 PG13。"
            "如果较高等级所需的证据看不清，保持已确认的较低级别，并设置 review_required=true；"
            "连基础内容也无法判断才使用 UNCERTAIN。可能未成年人的性化内容必须人工复核，勿推测年龄。"
            "只输出一个 JSON 对象：rating、review_required、evidence、counterevidence、limitations。"
            "evidence 和 counterevidence 均为对象数组，每项含 picture（如 Picture 12；无法定位时为空字符串）、"
            "observation（只写可见事实）、rule_id（VIS-01 至 VIS-06）、visibility（clear/occluded/uncertain）。"
            "至少记录一条可见事实。PG 可用 VIS-06 表示普通内容；看不清可用 VIS-05。"
            "不要编造秒数或看不到的动作。limitations 应写抽样帧、音频和可靠时间戳的限制。"
        )
        return (prompt,)


class BuildMetaTextAuditV0:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse_prompt": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audit_prompt",)
    FUNCTION = "build"
    CATEGORY = "video/review"

    def build(self, reverse_prompt):
        standard = _meta_standard_v0()
        prompt = (
            f"审核标准版本 {standard['version']}。只审核下面反推提示词明确写出的文字，不推断原视频。"
            f"内部描述性等级：{_meta_rating_guide(standard)}。"
            f"规则：{_meta_rule_guide(standard)}。"
            "不要把服装、身体轮廓、递物或靠近自动解释成裸体或性行为。"
            "PG13 必须有文字明确写出的独立性化镜头或动作；女仆装、丝袜、腮红、"
            "身体前倾、双臂交叉、挥拳、有人伸手靠近等文字组合仍不能单独构成 VIS-01，"
            "只有这些描述时判 PG，并以 VIS-06 记录。"
            "只输出一个 JSON 对象：rating、review_required、evidence、counterevidence、limitations。"
            "evidence 和 counterevidence 为对象数组，每项含 picture（文本审核固定为空字符串）、"
            "observation（简短引用文字含义）、rule_id（VIS-01 至 VIS-06）、visibility（clear/occluded/uncertain）。"
            "若文字没写出高一级所需内容，不能推断；若无法判断则选 UNCERTAIN。"
            "limitations 写明只审核反推文字，不能证明原视频安全。\n\n待审核文字：\n"
        )
        return (prompt + reverse_prompt,)


def _extract_json_object(raw_text):
    if not isinstance(raw_text, str):
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(raw_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


class MetaEvidenceNormalizeV0:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"raw_text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "rating")
    FUNCTION = "normalize"
    CATEGORY = "video/review"

    def normalize(self, raw_text):
        standard = _meta_standard_v0()
        ratings = {item["id"] for item in standard["rating_scale"]}
        rule_ids = {item["id"] for item in standard["rules"]}
        parsed = _extract_json_object(raw_text)
        rating = parsed.get("rating") if parsed else None
        if isinstance(rating, str):
            rating = rating.upper().replace("-", "").replace(" ", "")
        if rating not in ratings:
            parsed = {}
            rating = "UNCERTAIN"

        issues = []
        if not parsed:
            issues.append("模型未返回可解析的标准评级 JSON")

        def findings(field):
            value = parsed.get(field, [])
            if not isinstance(value, list):
                issues.append(f"{field} 不是对象数组")
                return []
            result = []
            for item in value[:12]:
                if not isinstance(item, dict):
                    issues.append(f"{field} 含非对象项")
                    continue
                rule_id = str(item.get("rule_id", ""))
                observation = str(item.get("observation", "")).strip()
                if rule_id not in rule_ids or not observation:
                    issues.append(f"{field} 缺少有效规则编号或观察")
                    continue
                visibility = item.get("visibility")
                if visibility not in {"clear", "occluded", "uncertain"}:
                    visibility = "uncertain"
                    issues.append(f"{field} 的可见性不明确")
                result.append({
                    "picture": str(item.get("picture") or "")[:80],
                    "observation": observation[:500],
                    "rule_id": rule_id,
                    "visibility": visibility,
                    "verification": "model_unverified",
                })
            return result

        evidence = findings("evidence")
        counterevidence = findings("counterevidence")
        if rating != "UNCERTAIN" and not evidence and not counterevidence:
            issues.append("缺少结构化画面或文本证据")
        required_rule = {"PG13": "VIS-01", "R": "VIS-02", "X": "VIS-03", "XXX": "VIS-04"}.get(rating)
        if required_rule and not any(
            item["rule_id"] == required_rule and item["visibility"] == "clear" for item in evidence
        ):
            issues.append(f"{rating} 缺少 {required_rule} 的清晰证据")

        normalized = {
            "standard_version": standard["version"],
            "rating": rating,
            "review_required": rating == "UNCERTAIN" or parsed.get("review_required") is True or bool(issues),
            "evidence": evidence,
            "counterevidence": counterevidence,
            "limitations": str(parsed.get("limitations") or "")[:500],
            "validation_issues": issues,
        }
        return (json.dumps(normalized, ensure_ascii=False), rating)


class MetaEvidenceReportV0:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_json": ("STRING", {"forceInput": True}),
                "prompt_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {"workflow_prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report_json",)
    FUNCTION = "build"
    CATEGORY = "video/review"

    def build(self, video_json, prompt_json, workflow_prompt=None):
        standard = _meta_standard_v0()

        video_sources = []
        visual_sampling = []
        if isinstance(workflow_prompt, dict):
            for node_id, node in workflow_prompt.items():
                if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
                    continue
                inputs = node["inputs"]
                if node.get("class_type") == "VHS_LoadVideo":
                    video_sources.append({
                        "node_id": node_id,
                        "filename": inputs.get("video"),
                        "force_rate": inputs.get("force_rate"),
                        "custom_height": inputs.get("custom_height"),
                    })
                if node.get("class_type") == "llama_cpp_instruct_adv" and inputs.get("inference_mode") == "video":
                    visual_sampling.append({
                        "node_id": node_id,
                        "max_frames": inputs.get("max_frames"),
                        "max_size": inputs.get("max_size"),
                    })

        def assessment(raw):
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                value = None
            if not isinstance(value, dict) or value.get("standard_version") != standard["version"]:
                return {"rating": "UNCERTAIN", "review_required": True, "evidence": [],
                        "counterevidence": [], "validation_issues": ["输入无效或标准版本不一致"]}
            return value

        video = assessment(video_json)
        prompt = assessment(prompt_json)
        video_rating = video["rating"]
        prompt_rating = prompt["rating"]
        disagreement = video_rating != prompt_rating and "UNCERTAIN" not in (video_rating, prompt_rating)
        review_required = (
            video.get("review_required") is not False
            or prompt.get("review_required") is not False
            or disagreement
            or video_rating != "PG"
        )
        risk_signal = {
            "PG": "no_clear_visual_signal_in_sample",
            "PG13": "possible_sexual_suggestion_review",
            "R": "visible_nudity_claim_review",
            "X": "visible_sexual_activity_claim_review",
            "XXX": "visible_detailed_sexual_activity_claim_review",
            "UNCERTAIN": "insufficient_visual_evidence",
        }.get(video_rating, "insufficient_visual_evidence")
        report = {
            "report_type": "pre_submission_video_evidence_draft",
            "standard_version": standard["version"],
            "scope": standard["scope"],
            "policy_references": standard["policy_references"],
            "source_video": video_sources,
            "visual_sampling": visual_sampling,
            "video_rating": video_rating,
            "prompt_rating": prompt_rating,
            "rating_disagreement": disagreement,
            "risk_signal": risk_signal,
            "review_required": review_required,
            "video_findings": video.get("evidence", []),
            "video_counterevidence": video.get("counterevidence", []),
            "prompt_findings": prompt.get("evidence", []),
            "prompt_counterevidence": prompt.get("counterevidence", []),
            "validation_issues": {"video": video.get("validation_issues", []),
                                  "prompt": prompt.get("validation_issues", [])},
            "coverage_limitations": ["仅覆盖抽样静帧；模型的 Picture 编号和描述均须对照原视频核查",
                                     "未审核音频、广告文案、落地页、定向或账户状态"],
            "model_limitations": {"video": video.get("limitations", ""),
                                  "prompt": prompt.get("limitations", "")},
            "approval_probability_percent": None,
            "meta_outcome": {"status": "unknown", "ad_id": None, "review_reason": None},
        }
        return (json.dumps(report, ensure_ascii=False),)


NODE_CLASS_MAPPINGS = {
    "NSFWJSONNormalize": NSFWJSONNormalize,
    "BuildNSFWPromptAudit": BuildNSFWPromptAudit,
    "BuildNSFWPromptAuditV2": BuildNSFWPromptAuditV2,
    "CombineNSFWAssessments": CombineNSFWAssessments,
    "CivitaiJSONNormalize": CivitaiJSONNormalize,
    "BuildCivitaiPromptAudit": BuildCivitaiPromptAudit,
    "BuildCivitaiImagePromptAudit": BuildCivitaiImagePromptAudit,
    "CombineCivitaiAssessments": CombineCivitaiAssessments,
    "MetaSexualRiskScore": MetaSexualRiskScore,
    "MetaApprovalEstimate": MetaApprovalEstimate,
    "MetaImageSexualRiskScore": MetaImageSexualRiskScore,
    "MetaImageApprovalEstimate": MetaImageApprovalEstimate,
    "BuildMetaVideoAuditV0": BuildMetaVideoAuditV0,
    "BuildMetaTextAuditV0": BuildMetaTextAuditV0,
    "MetaEvidenceNormalizeV0": MetaEvidenceNormalizeV0,
    "MetaEvidenceReportV0": MetaEvidenceReportV0,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWJSONNormalize": "NSFW JSON 校验",
    "BuildNSFWPromptAudit": "构建提示词审计输入",
    "BuildNSFWPromptAuditV2": "构建提示词审计输入 V2",
    "CombineNSFWAssessments": "合并视频与提示词判定",
    "CivitaiJSONNormalize": "Civitai JSON 校验",
    "BuildCivitaiPromptAudit": "构建 Civitai 提示词审计输入",
    "BuildCivitaiImagePromptAudit": "构建 Civitai 图片提示词审计输入",
    "CombineCivitaiAssessments": "合并 Civitai 视频与提示词判定",
    "MetaSexualRiskScore": "Meta 素材性内容参考分",
    "MetaApprovalEstimate": "Meta 视频画面过审概率估计",
    "MetaImageSexualRiskScore": "Meta 图片性内容参考分",
    "MetaImageApprovalEstimate": "Meta 图片画面过审概率估计",
    "BuildMetaVideoAuditV0": "构建 Meta v0 视频审核提示词",
    "BuildMetaTextAuditV0": "构建 Meta v0 反推文字审核提示词",
    "MetaEvidenceNormalizeV0": "Meta v0 证据 JSON 校验",
    "MetaEvidenceReportV0": "Meta v0 视频证据报告",
}

NODE_CLASS_MAPPINGS.update(IMAGE_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SHIELDGEMMA_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SHIELDGEMMA_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SAFEWATCH_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SAFEWATCH_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(APPROVAL_RISK_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(APPROVAL_RISK_NODE_DISPLAY_NAME_MAPPINGS)
NODE_CLASS_MAPPINGS.update(AGE_SAFETY_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(AGE_SAFETY_NODE_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = "./web"
