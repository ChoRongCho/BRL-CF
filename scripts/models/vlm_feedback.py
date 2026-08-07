"""GPT-4o 기반 VLM answer model."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL_NAME = "gpt-4o"
IMAGE_PLACEHOLDER = "<IMAGE_URL_OR_BASE64_DATA_URL>"
VLM_IMAGE_ROOT = Path(__file__).resolve().parents[1] / "vlm_images"
PROMPT = """Inspect the snapshot and determine whether the following fact is true.
{action_context}Fact: {target_fact}
Answer with only true or false."""
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_snapshot(
    domain: str,
    initial_state: str | Path,
    step: int,
) -> Path:
    """현재 domain/scene/step에 해당하는 snapshot 경로를 반환한다."""
    scene_number_match = re.search(r"(\d+)$", Path(initial_state).stem)
    if scene_number_match is None:
        raise ValueError(f"Cannot determine scene number from {initial_state!s}")

    scene_number = int(scene_number_match.group(1))
    scene_dir = VLM_IMAGE_ROOT / domain / f"scene{scene_number}"
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"VLM snapshot directory not found: {scene_dir}")

    expected_stems = {f"step{step}", f"step_{step}", f"snapshot{step}", f"snapshot_{step}"}
    matches = sorted(
        path
        for path in scene_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        and path.stem.lower() in expected_stems
    )
    if not matches:
        raise FileNotFoundError(
            f"No snapshot for step {step} in {scene_dir}. "
            f"Use a filename such as step{step}.jpg."
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple snapshots found for step {step}: {matches}")
    return matches[0]


def image_to_data_url(image_path: str | Path) -> str:
    """로컬 이미지 파일을 OpenAI image_url용 Base64 data URL로 변환한다."""
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def answer_question(
    question: str,
    image: str | Path = IMAGE_PLACEHOLDER,
    *,
    client: OpenAI | None = None,
) -> str:
    """이미지와 질문을 GPT-4o에 전달하고 답변 텍스트를 반환한다.

    ``image``에는 공개 이미지 URL 또는
    ``data:image/jpeg;base64,...`` 형식의 data URL을 전달한다.
    """
    if str(image) == IMAGE_PLACEHOLDER:
        raise ValueError(
            "IMAGE_PLACEHOLDER를 실제 이미지 URL 또는 Base64 data URL로 교체하세요."
        )

    image_value = str(image)
    if not image_value.startswith(("http://", "https://", "data:")):
        image_value = image_to_data_url(image_value)

    load_dotenv()
    openai_client = client or OpenAI()
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_value},
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content or ""


def answer_fact_question(
    target_fact: str,
    image: str | Path = IMAGE_PLACEHOLDER,
    *,
    action_name: str | None = None,
    client: OpenAI | None = None,
) -> bool:
    """이미지를 근거로 symbolic fact의 참/거짓을 판정한다."""
    action_context = (
        f"The most recent robot action was: {action_name}.\n"
        if action_name
        else ""
    )
    prompt = PROMPT.format(
        action_context=action_context,
        target_fact=target_fact,
    )
    answer = answer_question(prompt, image, client=client).strip().lower()

    if answer in {"true", "t"}:
        return True
    if answer in {"false", "f"}:
        return False
    raise ValueError(f"GPT-4o returned an invalid boolean answer: {answer!r}")


if __name__ == "__main__":
    answer = answer_question(
        question="이 이미지를 보고 질문에 답해주세요.",
        image=IMAGE_PLACEHOLDER,  # TODO: 실제 이미지 URL 또는 Base64 data URL
    )
    print(answer)
