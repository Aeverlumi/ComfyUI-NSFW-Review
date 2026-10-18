import json


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


NODE_CLASS_MAPPINGS = {
    "NSFWJSONNormalize": NSFWJSONNormalize,
    "BuildNSFWPromptAudit": BuildNSFWPromptAudit,
    "BuildNSFWPromptAuditV2": BuildNSFWPromptAuditV2,
    "CombineNSFWAssessments": CombineNSFWAssessments,
    "CivitaiJSONNormalize": CivitaiJSONNormalize,
    "BuildCivitaiPromptAudit": BuildCivitaiPromptAudit,
    "CombineCivitaiAssessments": CombineCivitaiAssessments,
    "MetaSexualRiskScore": MetaSexualRiskScore,
    "MetaApprovalEstimate": MetaApprovalEstimate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWJSONNormalize": "NSFW JSON 校验",
    "BuildNSFWPromptAudit": "构建提示词审计输入",
    "BuildNSFWPromptAuditV2": "构建提示词审计输入 V2",
    "CombineNSFWAssessments": "合并视频与提示词判定",
    "CivitaiJSONNormalize": "Civitai JSON 校验",
    "BuildCivitaiPromptAudit": "构建 Civitai 提示词审计输入",
    "CombineCivitaiAssessments": "合并 Civitai 视频与提示词判定",
    "MetaSexualRiskScore": "Meta 素材性内容参考分",
    "MetaApprovalEstimate": "Meta 视频画面过审概率估计",
}
