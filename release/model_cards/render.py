"""Render README.md + LICENSE/NOTICE bundle into a staged HF dir.

Run after hf_stage_and_scrub.py so the upload ships a complete,
license-compliant repo in one shot.
"""
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from license_stanzas import BY_PRESET  # noqa: E402


def render(preset: str, stage_dir: str, base_model: str, layer_idx: str,
           this_repo: str, pair_repo: str, role: str,
           training_fve: str = "—") -> None:
    spec = BY_PRESET[preset]
    stage = Path(stage_dir)
    role_desc = {"av": "AV (activation verbalizer, vector → text)",
                 "ar": "AR (activation reconstructor, text → vector)"}[role]

    tmpl = (HERE / "README_template.md").read_text()
    readme = tmpl.format(
        LICENSE_TAG=spec["license_tag"],
        BASE_MODEL=base_model,
        BUILT_WITH_BANNER=spec["built_with_banner"],
        DISPLAY_NAME=this_repo.split("/")[-1],
        ROLE_DESC=role_desc,
        PAIR_REPO=pair_repo,
        LAYER_IDX=layer_idx,
        TRAINING_FVE=training_fve,
        LICENSE_STANZA=spec["license_stanza"],
    )
    (stage / "README.md").write_text(readme)

    if "notice_src" in spec:
        (stage / "NOTICE").write_text((HERE / spec["notice_src"]).read_text())

    if "license_url" in spec:
        (stage / "LICENSE").write_bytes(
            urllib.request.urlopen(spec["license_url"]).read())

    if "use_policy_url" in spec:
        (stage / "USE_POLICY.md").write_bytes(
            urllib.request.urlopen(spec["use_policy_url"]).read())

    print(f"rendered model card + legal bundle into {stage}")


if __name__ == "__main__":
    render(*sys.argv[1:])
