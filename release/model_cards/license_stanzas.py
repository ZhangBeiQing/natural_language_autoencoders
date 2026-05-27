"""Per-base-model license stanzas + metadata for NLA model cards."""

LLAMA = dict(
    license_tag="llama3.3",
    built_with_banner="**Built with Llama**",
    license_stanza=(
        "This model is a derivative of Llama 3.3 and is distributed under the "
        "[Llama 3.3 Community License Agreement]"
        "(https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE). "
        "By using this model you agree to the license and the accompanying "
        "[Acceptable Use Policy]"
        "(https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/USE_POLICY.md). "
        "See `LICENSE`, `USE_POLICY.md`, and `NOTICE` in this repository."
    ),
    bundle=["LICENSE", "USE_POLICY.md", "NOTICE"],
    notice_src="NOTICE.llama",
    license_url="https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_3/LICENSE",
    use_policy_url="https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_3/USE_POLICY.md",
)

GEMMA = dict(
    license_tag="gemma",
    built_with_banner="",
    license_stanza=(
        "This model is a derivative of Gemma 3 and is provided under and subject "
        "to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). By "
        "using this model you agree to those terms and the "
        "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). "
        "See `NOTICE` in this repository."
    ),
    bundle=["NOTICE"],
    notice_src="NOTICE.gemma",
)

QWEN = dict(
    license_tag="apache-2.0",
    built_with_banner="",
    license_stanza=(
        "This model is fine-tuned from Qwen/Qwen2.5-7B-Instruct and is "
        "distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). "
        "See `LICENSE` in this repository."
    ),
    bundle=["LICENSE"],
    license_url="https://www.apache.org/licenses/LICENSE-2.0.txt",
)

BY_PRESET = {
    "qwen7b": QWEN,
    "gemma12b": GEMMA,
    "gemma27b": GEMMA,
    "llama70b": LLAMA,
}
