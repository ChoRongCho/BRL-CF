from __future__ import annotations

import copy
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


SRC = Path("/home/changmin/Downloads/14_AIRLAB_BRL_박창민.pptx")
OUT = Path("14_AIRLAB_BRL_박창민_filled.pptx")

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("p", P_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("", PKG_REL_NS)

NS = {"p": P_NS, "a": A_NS}

EMU = 914400


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def clear_txbody(tx_body: ET.Element) -> None:
    for child in list(tx_body):
        tx_body.remove(child)


def make_run(text: str, size: int = 2200, bold: bool = False) -> ET.Element:
    run = ET.Element(qn(A_NS, "r"))
    rpr = ET.SubElement(run, qn(A_NS, "rPr"), {"lang": "ko-KR", "sz": str(size)})
    if bold:
        rpr.set("b", "1")
    ET.SubElement(rpr, qn(A_NS, "latin"), {"typeface": "Aptos"})
    ET.SubElement(rpr, qn(A_NS, "ea"), {"typeface": "맑은 고딕"})
    ET.SubElement(run, qn(A_NS, "t")).text = text
    return run


def make_para(text: str, level: int = 0, size: int = 2200, bullet: bool = True) -> ET.Element:
    para = ET.Element(qn(A_NS, "p"))
    ppr = ET.SubElement(para, qn(A_NS, "pPr"))
    if level:
        ppr.set("lvl", str(level))
    if bullet:
        ET.SubElement(ppr, qn(A_NS, "buChar"), {"char": "•"})
    else:
        ET.SubElement(ppr, qn(A_NS, "buNone"))
    para.append(make_run(text, size=size))
    return para


def replace_text(shape: ET.Element, lines: list[str], *, size: int = 2200, bullet: bool = True) -> None:
    tx_body = shape.find("p:txBody", NS)
    if tx_body is None:
        return
    body_pr = tx_body.find("a:bodyPr", NS)
    lst_style = tx_body.find("a:lstStyle", NS)
    clear_txbody(tx_body)
    if body_pr is not None:
        tx_body.append(copy.deepcopy(body_pr))
    else:
        tx_body.append(ET.Element(qn(A_NS, "bodyPr")))
    if lst_style is not None:
        tx_body.append(copy.deepcopy(lst_style))
    else:
        tx_body.append(ET.Element(qn(A_NS, "lstStyle")))
    for line in lines:
        if not line:
            tx_body.append(make_para("", size=size, bullet=False))
        elif line.startswith("  - "):
            tx_body.append(make_para(line[4:], level=1, size=max(size - 200, 1600), bullet=True))
        elif line.startswith("- "):
            tx_body.append(make_para(line[2:], level=0, size=size, bullet=bullet))
        else:
            tx_body.append(make_para(line, level=0, size=size, bullet=bullet))


def find_shape(root: ET.Element, name_prefix: str) -> ET.Element | None:
    for shape in root.findall(".//p:sp", NS):
        c_nv_pr = shape.find(".//p:cNvPr", NS)
        if c_nv_pr is not None and c_nv_pr.get("name", "").startswith(name_prefix):
            return shape
    return None


def max_shape_id(root: ET.Element) -> int:
    vals = []
    for c_nv_pr in root.findall(".//p:cNvPr", NS):
        try:
            vals.append(int(c_nv_pr.get("id", "0")))
        except ValueError:
            pass
    return max(vals or [1])


def add_textbox(
    root: ET.Element,
    name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[str],
    *,
    size: int = 2100,
    bullet: bool = True,
) -> None:
    sp_tree = root.find(".//p:cSld/p:spTree", NS)
    if sp_tree is None:
        return
    sp = ET.Element(qn(P_NS, "sp"))
    nv_sp_pr = ET.SubElement(sp, qn(P_NS, "nvSpPr"))
    ET.SubElement(nv_sp_pr, qn(P_NS, "cNvPr"), {"id": str(max_shape_id(root) + 1), "name": name})
    ET.SubElement(nv_sp_pr, qn(P_NS, "cNvSpPr"), {"txBox": "1"})
    ET.SubElement(nv_sp_pr, qn(P_NS, "nvPr"))
    sp_pr = ET.SubElement(sp, qn(P_NS, "spPr"))
    xfrm = ET.SubElement(sp_pr, qn(A_NS, "xfrm"))
    ET.SubElement(xfrm, qn(A_NS, "off"), {"x": str(int(x * EMU)), "y": str(int(y * EMU))})
    ET.SubElement(xfrm, qn(A_NS, "ext"), {"cx": str(int(w * EMU)), "cy": str(int(h * EMU))})
    prst = ET.SubElement(sp_pr, qn(A_NS, "prstGeom"), {"prst": "rect"})
    ET.SubElement(prst, qn(A_NS, "avLst"))
    tx_body = ET.SubElement(sp, qn(P_NS, "txBody"))
    ET.SubElement(tx_body, qn(A_NS, "bodyPr"), {"wrap": "square", "lIns": "91440", "tIns": "45720", "rIns": "91440", "bIns": "45720"})
    ET.SubElement(tx_body, qn(A_NS, "lstStyle"))
    for line in lines:
        tx_body.append(make_para(line[2:] if line.startswith("- ") else line, size=size, bullet=bullet))
    sp_tree.append(sp)


def remove_generated_body(root: ET.Element) -> None:
    sp_tree = root.find(".//p:cSld/p:spTree", NS)
    if sp_tree is None:
        return
    for shape in list(sp_tree.findall("p:sp", NS)):
        c_nv_pr = shape.find(".//p:cNvPr", NS)
        if c_nv_pr is not None and c_nv_pr.get("name", "").startswith("Generated Body"):
            sp_tree.remove(shape)


SLIDES = {
    1: {
        "title": ["BRL CF 논문 방향성"],
        "subtitle": ["POMDP planning + mismatch-based feedback trigger"],
        "footer": ["Changmin Park / AI Robotics Lab, Sogang University"],
    },
    2: {
        "title": ["현재 상황: 기존 유저 스터디의 한계"],
        "body": [
            "- N=12, 도메인별 5조건, 총 10조건 수행",
            "- 참가자는 3~4분 로봇 영상을 보며 계속 행동 적절성을 확인",
            "- 조건별 fatigue / workload 차이가 명확하지 않음",
            "- Ours1, Ours2 차이가 해석을 복잡하게 만듦",
            "- 핵심 문제는 표본 수만이 아니라 실험 과제가 시스템 취지와 어긋난 점",
            "- 따라서 기존 데이터를 그대로 밀기보다 claim, task, condition을 다시 맞춰야 함",
        ],
    },
    3: {
        "title": ["핵심 문제 정의"],
        "body": [
            "- 시스템 목적: 사용자가 계속 감시하는 것이 아니라 필요한 순간에만 feedback 요청",
            "- 기존 과제는 '상시 감시'를 요구해 interruption / feedback efficiency를 보기 어려움",
            "- 일부 데이터 오염 가능성, N=6 재분석에서도 명확한 결론 부족",
            "- 따라서 단순 확장보다 claim과 실험 설계를 다시 맞추는 것이 필요",
            "- 새 실험은 workload 일반론보다 '주 작업 중 feedback request의 방해와 효율'에 초점",
        ],
    },
    4: {
        "title": ["논문 전략 판단"],
        "body": [
            "- 시스템 논문 1편 + 후속 N=30 논문으로 나누는 전략은 위험",
            "- 시스템 논문은 human evidence가 약하다는 비판 가능",
            "- 후속 논문은 '더 많은 사람으로 검증' 수준에 머물 수 있음",
            "- 더 자연스러운 구조: 시스템 제안 + simulation/ablation + N=30 user study를 한 논문에 통합",
        ],
        "boxes": {
            "사각형: 둥근 모서리 4": ["분리 전략", "시스템 기여 1편"],
            "사각형: 둥근 모서리 5": ["후속 전략", "HRI 기여 1편", "N>=30"],
            "사각형: 둥근 모서리 8": ["위험", "human evidence 약함", "후속 논문 독립성 약함"],
            "사각형: 둥근 모서리 9": ["권장", "시스템 + simulation", "+ N=30 user study 통합"],
        },
    },
    5: {
        "title": ["시스템의 핵심 정체성"],
        "body": [
            "- 로봇은 task 수행을 위한 state, knowledge, transition model을 가짐",
            "- 실제 observation이 transition model의 예측 분포와 다르면 mismatch를 감지",
            "- 질문 이유는 단순 uncertainty가 아니라 '예상한 세계'와 '관측된 세계'의 불일치",
            "- Feedback은 belief / knowledge update로 들어가고 replanning에 반영",
            "- 핵심은 사람을 planner의 외부 oracle로 쓰는 것이 아니라 closed-loop correction source로 쓰는 점",
        ],
    },
    6: {
        "title": ["Proposed Loop"],
        "body": [
            "- POMDP planning -> action execution -> observation",
            "- belief / uncertainty update",
            "- observation-transition mismatch 또는 uncertainty threshold check",
            "- human feedback request",
            "- knowledge / belief update -> replanning",
            "- 질문을 POMDP action space에 넣기보다 별도 trigger layer로 처리",
        ],
    },
    7: {
        "title": ["KnowNo와의 차이"],
        "body": [
            "- KnowNo: action-level uncertainty",
            "  - '지금 어떤 action을 해야 하는가?'가 불확실할 때 도움 요청",
            "- Ours: state / knowledge-level uncertainty",
            "  - '내가 가진 상태나 지식 중 무엇이 실제 세계와 맞지 않는가?'를 확인",
            "- 한 번의 feedback이 현재 action뿐 아니라 이후 여러 planning step에 재사용 가능",
        ],
    },
    8: {
        "title": ["가능한 Contribution"],
        "body": [
            "- POMDP planning과 feedback trigger의 결합",
            "- Observation-transition mismatch 기반 질문",
            "- State / knowledge-level feedback으로 action-level baseline과 구분",
            "- Human feedback을 belief / knowledge update에 반영하는 closed-loop replanning",
            "- Oracle / noisy / biased feedback 조건에서 robustness 평가 가능",
        ],
    },
    9: {
        "title": ["새 N=30 유저 스터디 방향"],
        "body": [
            "- 조건을 5개에서 3개로 축소: All vs Ours vs KnowNo",
            "- Main task 수행 중 tablet feedback이 sub task로 interrupt",
            "- 기존 '동영상 감시'가 아니라 system-initiated feedback의 방해와 효율을 측정",
            "- Ours1/Ours2 차이는 user study보다 simulation / ablation에서 다루는 편이 적절",
            "- N=30은 단순히 power를 키우는 것이 아니라 재설계된 HRI claim을 검증하는 핵심 평가",
        ],
    },
    10: {
        "title": ["새 Claim과 측정값"],
        "body": [
            "- Claim: 사용자의 주 작업을 덜 방해하면서 필요한 feedback을 얻는다",
            "- 주 작업 성능 및 수행 시간",
            "- feedback 정확도 및 응답 시간",
            "- 질문 개수, 질문 후 주 작업 복귀 시간",
            "- perceived workload, interruption burden, timing appropriateness, trust / usefulness",
        ],
    },
    11: {
        "title": ["Hypotheses"],
        "body": [
            "- H1. Ours는 All보다 주 작업 방해와 perceived workload를 줄인다",
            "- H2. Ours는 All보다 적은 질문으로 비슷한 feedback 품질을 유지한다",
            "- H3. Ours는 KnowNo보다 planning에 재사용 가능한 state/knowledge-level feedback을 더 효율적으로 얻는다",
            "- H4. 사용자는 Ours의 질문 타이밍을 더 적절하다고 인식한다",
        ],
    },
    12: {
        "title": ["교수님께 말할 요지"],
        "body": [
            "- 기존 실험은 N 부족보다 실험 과제가 시스템 취지와 맞지 않는 문제가 큼",
            "- N=30을 한다면 기존 설계 확장이 아니라 main task 중 system-initiated query 구조로 재설계",
            "- 조건은 All / Ours / KnowNo로 줄이고, Ours 내부 차이는 ablation에서 검증",
            "- 최종 논문은 시스템 + simulation/ablation + N=30 user study로 통합하는 방향이 가장 탄탄",
        ],
    },
    13: {
        "title": ["Backup: 왜 논문을 나누기 어려운가"],
        "body": [
            "- 시스템 논문만 먼저 내면 'human-facing system인데 human evidence가 약하다'는 약점이 남음",
            "- 후속 N=30 논문은 독립적인 연구 질문 없이 단순 validation처럼 보일 수 있음",
            "- 반대로 한 논문에 묶으면 algorithmic novelty와 HRI relevance가 서로 보강됨",
            "- Simulation/ablation은 mechanism을 설명하고, user study는 실제 사용 맥락의 burden과 효율을 설명",
            "- 결론: N=30을 한다면 후속 논문이 아니라 현재 논문의 핵심 평가로 넣는 편이 자연스러움",
        ],
    },
    14: {
        "title": ["Backup: 새 실험 프로토콜"],
        "body": [
            "- 참가자는 주 작업을 수행하고, 로봇/시스템 질문은 태블릿 sub task로 중간에 등장",
            "- 각 조건에서 동일한 도메인 난이도와 비슷한 총 task duration을 유지",
            "- 질문이 들어오면 참가자는 잠시 feedback을 제공한 뒤 주 작업으로 복귀",
            "- 로그: 질문 발생 시점, 응답 시작/종료, 정답 여부, 주 작업 중단/복귀 시점",
            "- 설문: workload, interruption burden, resumption difficulty, timing appropriateness, trust/usefulness",
        ],
    },
    15: {
        "title": ["Backup: 조건 정의"],
        "body": [
            "- All: 가능한 ambiguity마다 자주 묻는 high-query baseline",
            "- Ours: belief uncertainty와 observation-transition mismatch가 threshold를 넘을 때 state/knowledge 질문",
            "- KnowNo: 다음 action 후보가 불확실할 때 action selection 질문",
            "- 비교 축 1: 질문 수와 feedback quality의 trade-off",
            "- 비교 축 2: action-level 질문과 state/knowledge-level 질문의 장기적 planning 효과",
            "- 비교 축 3: 주 작업 중 interruption burden과 질문 타이밍의 적절성",
        ],
    },
    16: {
        "title": ["Backup: 측정값과 분석 계획"],
        "body": [
            "- Objective: main task score/time, feedback accuracy, response time, resumption time, number of queries",
            "- Subjective: NASA-TLX 또는 short workload scale, interruption burden, switching/resumption difficulty",
            "- System: task success, planning length, replanning count, final belief/knowledge correction accuracy",
            "- 분석은 within-subject 조건 비교를 기본으로 하고, 순서 효과를 counterbalancing",
            "- 핵심 결과는 '질문을 줄였는가'만이 아니라 '줄인 질문이 planning에 필요한 정보를 유지했는가'",
        ],
    },
    17: {
        "title": ["Backup: 관련 연구에서 가져올 근거"],
        "body": [
            "- Banerjee et al.: interruptibility-aware robot behavior는 주 작업 성능보다 timing appropriateness와 robot task efficiency에서 효과가 날 수 있음",
            "- Dahiya et al.: interruption은 main task performance보다 perceived workload와 task switching difficulty에 더 민감하게 나타날 수 있음",
            "- KnowNo: action-level ask-for-help baseline으로 적합하지만, state/knowledge correction과는 질문 단위가 다름",
            "- 따라서 우리 실험은 main task success만 보지 말고 interruption quality와 feedback utility를 함께 봐야 함",
        ],
    },
    18: {
        "title": ["Backup: 논문 한 문장 정리"],
        "body": [
            "- 이 논문은 POMDP 기반 task planning 과정에서 생기는 belief uncertainty와 observation-transition mismatch를 감시한다",
            "- 시스템은 필요한 순간에만 사람에게 state/knowledge-level feedback을 요청한다",
            "- 받은 feedback은 belief/knowledge update와 replanning으로 닫힌 고리를 만든다",
            "- 평가는 simulation/ablation으로 trigger와 feedback model의 효과를 보이고, N=30 user study로 주 작업 중 방해와 효율을 검증한다",
            "- 최종 claim: 적은 질문으로 planning에 필요한 feedback을 유지하면서 사용자의 주 작업 방해를 줄인다",
        ],
    },
}


def patch_slide(xml_bytes: bytes, slide_no: int) -> bytes:
    root = ET.fromstring(xml_bytes)
    data = SLIDES[slide_no]
    remove_generated_body(root)

    title = find_shape(root, "Title") or find_shape(root, "제목")
    if title is not None:
        replace_text(title, data["title"], size=3000, bullet=False)
    elif "title" in data:
        add_textbox(root, f"Generated Title {slide_no}", 0.55, 0.28, 12.0, 0.65, data["title"], size=3000, bullet=False)

    if slide_no == 1:
        subtitle = find_shape(root, "Subtitle")
        footer = find_shape(root, "Rectangle")
        if subtitle is not None:
            replace_text(subtitle, data["subtitle"], size=2200, bullet=False)
        if footer is not None:
            replace_text(footer, data["footer"], size=1400, bullet=False)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    if slide_no == 4:
        for name, lines in data["boxes"].items():
            shape = find_shape(root, name)
            if shape is not None:
                replace_text(shape, lines, size=1700, bullet=False)
        slide_num = find_shape(root, "슬라이드 번호")
        if slide_num is not None:
            replace_text(slide_num, [str(slide_no - 1)], size=1200, bullet=False)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    body = find_shape(root, "내용")
    if body is not None:
        replace_text(body, data["body"], size=1950 if slide_no >= 13 else 2050, bullet=True)
    else:
        add_textbox(root, f"Generated Body {slide_no}", 0.78, 1.35, 11.85, 5.4, data["body"], size=1950 if slide_no >= 13 else 2050, bullet=True)

    slide_num = find_shape(root, "슬라이드 번호")
    if slide_num is not None:
        replace_text(slide_num, [str(slide_no - 1)], size=1200, bullet=False)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def add_slide_to_presentation(xml_bytes: bytes, slide_no: int, rel_id: str) -> bytes:
    root = ET.fromstring(xml_bytes)
    sld_id_lst = root.find("p:sldIdLst", NS)
    if sld_id_lst is None:
        raise RuntimeError("presentation.xml has no sldIdLst")
    max_id = max(int(el.get("id", "255")) for el in sld_id_lst.findall("p:sldId", NS))
    ET.SubElement(sld_id_lst, qn(P_NS, "sldId"), {"id": str(max_id + 1), qn(R_NS, "id"): rel_id})
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def add_presentation_rel(xml_bytes: bytes, slide_no: int, rel_id: str) -> bytes:
    root = ET.fromstring(xml_bytes)
    ET.SubElement(
        root,
        qn(PKG_REL_NS, "Relationship"),
        {
            "Id": rel_id,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
            "Target": f"slides/slide{slide_no}.xml",
        },
    )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def add_content_type(xml_bytes: bytes, slide_no: int) -> bytes:
    root = ET.fromstring(xml_bytes)
    part_name = f"/ppt/slides/slide{slide_no}.xml"
    exists = any(el.get("PartName") == part_name for el in root.findall(f"{{{CT_NS}}}Override"))
    if not exists:
        ET.SubElement(
            root,
            qn(CT_NS, "Override"),
            {
                "PartName": part_name,
                "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
            },
        )
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    data = data.replace(
        b'<ns0:Types xmlns:ns0="http://schemas.openxmlformats.org/package/2006/content-types"',
        b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"',
    )
    data = data.replace(b"</ns0:Types>", b"</Types>")
    data = data.replace(b"ns0:", b"")
    return data


def main() -> None:
    slide_re = re.compile(r"ppt/slides/slide(\d+)\.xml$")
    extra_slides = [n for n in sorted(SLIDES) if n > 12]
    with ZipFile(SRC, "r") as zin:
        entries = {item.filename: (item, zin.read(item.filename)) for item in zin.infolist()}
        template_slide = entries["ppt/slides/slide12.xml"][1]
        template_rels = entries["ppt/slides/_rels/slide12.xml.rels"][1]

    for slide_no in extra_slides:
        rel_id = f"rId{slide_no + 6}"
        entries["ppt/presentation.xml"] = (
            entries["ppt/presentation.xml"][0],
            add_slide_to_presentation(entries["ppt/presentation.xml"][1], slide_no, rel_id),
        )
        entries["ppt/_rels/presentation.xml.rels"] = (
            entries["ppt/_rels/presentation.xml.rels"][0],
            add_presentation_rel(entries["ppt/_rels/presentation.xml.rels"][1], slide_no, rel_id),
        )
        entries["[Content_Types].xml"] = (
            entries["[Content_Types].xml"][0],
            add_content_type(entries["[Content_Types].xml"][1], slide_no),
        )
        entries[f"ppt/slides/slide{slide_no}.xml"] = (None, template_slide)
        entries[f"ppt/slides/_rels/slide{slide_no}.xml.rels"] = (None, template_rels)

    with ZipFile(OUT, "w", ZIP_DEFLATED) as zout:
        for filename, (item, data) in entries.items():
            match = slide_re.match(filename)
            if match:
                slide_no = int(match.group(1))
                if slide_no in SLIDES:
                    data = patch_slide(data, slide_no)
            if item is None:
                zout.writestr(filename, data)
            else:
                zout.writestr(item, data)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
