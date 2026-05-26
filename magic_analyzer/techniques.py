"""마술 기법(슬레이트) 용어집 — 상세 설명 + 참고 튜토리얼 영상 링크.

에이전트의 explain_technique 도구가 이 사전을 조회한다. 참고 링크는 특정 영상이
삭제되어도 깨지지 않도록 YouTube '검색' URL을 쓴다(항상 동작 + 영상으로 설명).
"""

from __future__ import annotations

import urllib.parse


def _yt(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query)


# key는 매칭용 핵심어. aliases에 한/영 표기를 넣어 느슨하게 매칭한다.
TECHNIQUES: dict[str, dict] = {
    "french_drop": {
        "ko": "프렌치 드롭", "en": "French Drop",
        "aliases": ["french drop", "프렌치", "프렌치드롭", "프렌치 드롭"],
        "type": "coin/card 겸용",
        "desc": ("한 손(보통 오른손)으로 다른 손의 동전을 '집어가는 척'하지만, 실제로는 "
                 "원래 손(왼손)에 동전을 떨어뜨려 핑거팜으로 숨긴다. 관객은 가져가는 손에 "
                 "동전이 있다고 믿지만 빈손이다. 핵심 단서: 두 손이 잠깐 겹친 직후 '가져간' "
                 "손이 주먹을 쥔 채 자연스럽게 멀어지고, 원래 손은 살짝 오므려져 있다."),
        "cues": "두 손 접촉 → 한 손 주먹 쥐고 이동(FAST/VANISH), 원손 손가락 살짝 말림(GRAB)",
        "query": "french drop sleight tutorial slow motion",
    },
    "classic_palm": {
        "ko": "클래식 팜", "en": "Classic Palm",
        "aliases": ["classic palm", "클래식 팜", "팜", "palm", "palming"],
        "type": "coin/card 겸용",
        "desc": ("손바닥 근육으로 동전(또는 카드)을 눌러 쥐어, 손을 자연스럽게 편 것처럼 "
                 "보이면서도 물건을 숨긴다. 손이 비어 보이지만 손바닥 안에 물체가 있다. "
                 "단서: 손가락은 자연스럽게 펴져 있는데 손바닥이 미묘하게 굽어 있고, 손을 "
                 "뒤집거나 손바닥을 정면으로 보여주길 피한다."),
        "cues": "펼친 손이 갑자기 살짝 오므라듦(GRAB), 손바닥 노출 회피",
        "query": "classic palm coin tutorial",
    },
    "finger_palm": {
        "ko": "핑거 팜", "en": "Finger Palm",
        "aliases": ["finger palm", "핑거 팜", "핑거팜", "핑거 팔름", "finger palming"],
        "type": "coin 주로",
        "desc": ("손가락(약지·중지) 안쪽 마디에 동전을 가볍게 올려 자연스럽게 쥔다. "
                 "클래식 팜보다 편하고 손이 더 자연스러워 보인다. 프렌치 드롭 후 동전을 "
                 "이 위치에 숨기는 경우가 많다."),
        "cues": "손가락이 살짝 말린 채 유지, 손등을 관객 쪽으로",
        "query": "finger palm coin tutorial",
    },
    "thumb_palm": {
        "ko": "썸 팜", "en": "Thumb Palm",
        "aliases": ["thumb palm", "썸 팜", "썸팜", "thumb palming"],
        "type": "coin 주로",
        "desc": ("엄지와 검지 사이 갈퀴(웹) 부분에 동전을 끼워 숨긴다. 손을 펴도 동전이 "
                 "엄지 안쪽에 가려 보이지 않는다. 동전을 '던지는 척'하며 사라지게 할 때 자주 쓴다."),
        "cues": "엄지 안쪽으로 빠르게 동전 이동(FAST), 던지는 제스처 후 빈손",
        "query": "thumb palm coin vanish tutorial",
    },
    "double_lift": {
        "ko": "더블 리프트", "en": "Double Lift",
        "aliases": ["double lift", "더블 리프트", "더블리프트", "double turnover"],
        "type": "card",
        "desc": ("덱 맨 위 카드 2장을 1장인 것처럼 정확히 정렬해 한 번에 뒤집어 보여준다. "
                 "관객은 맨 위 카드를 봤다고 믿지만 실제로는 둘째 카드다. 앰비셔스 카드 등 "
                 "수많은 루틴의 기본기. 단서: 카드를 뒤집기 전 엄지로 살짝 '겟 레디(2장 분리)'."),
        "cues": "덱 위에서 손가락 미세 정렬 후 1장처럼 뒤집기, 빠른 손동작(FAST)",
        "query": "double lift card sleight tutorial",
    },
    "pass": {
        "ko": "패스(클래식 패스)", "en": "Classic Pass",
        "aliases": ["pass", "패스", "classic pass", "클래식 패스", "shift"],
        "type": "card",
        "desc": ("덱을 두 덩어리로 나눈 뒤, 순간적으로 위·아래 덩어리의 위치를 몰래 바꾼다. "
                 "관객이 끼워 넣은 카드를 맨 위로 비밀리에 올리는 데 쓴다. 매우 빠르고 작은 "
                 "동작이라 영상에서도 한두 프레임에만 보인다."),
        "cues": "덱을 잡은 양손이 순간적으로 빠르게 움직임(FAST), 손이 잠깐 덱을 덮음(CONTACT)",
        "query": "classic pass card magic tutorial slow motion",
    },
    "shuttle_pass": {
        "ko": "셔틀 패스", "en": "Shuttle Pass",
        "aliases": ["shuttle pass", "셔틀 패스", "셔틀패스", "shuttle"],
        "type": "coin",
        "desc": ("한 손에서 다른 손으로 동전을 '건네는 척'하지만 실제로는 원래 손에 팜으로 "
                 "남긴다. 두 손을 오가며 동전 개수를 속이는 코인 루틴의 핵심."),
        "cues": "두 손이 가까워지며 건네는 제스처(CONTACT) 직후 받은 손이 빈손",
        "query": "shuttle pass coin magic tutorial",
    },
    "retention_vanish": {
        "ko": "리텐션 베니시", "en": "Retention Vanish",
        "aliases": ["retention vanish", "retention", "리텐션", "리텐션 베니시"],
        "type": "coin",
        "desc": ("동전을 다른 손에 쥐여주는 '잔상'을 남기며 시각적으로 강하게 속인다. 관객은 "
                 "동전이 손에 들어가는 것을 '봤다'고 느끼지만 실제로는 원래 손에 남는다. "
                 "가장 시각적으로 깨끗한 코인 베니시 중 하나."),
        "cues": "동전을 쥐여주는 순간 잠깐 보였다가 손 닫힌 뒤 사라짐(VANISH)",
        "query": "retention vanish coin tutorial",
    },
    "top_change": {
        "ko": "탑 체인지", "en": "Top Change",
        "aliases": ["top change", "탑 체인지", "탑체인지"],
        "type": "card",
        "desc": ("손에 든 카드를 덱 맨 위 카드와 순간적으로 바꿔치기한다. 시선이 분산된 "
                 "순간(미스디렉션)에 이뤄져 거의 보이지 않는다."),
        "cues": "카드 든 손이 덱을 스치는 순간 빠른 교체(FAST), 시선 분산 타이밍",
        "query": "top change card sleight tutorial",
    },
}


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ")


def lookup(name: str) -> dict | None:
    """기법명(한/영, 느슨)으로 용어집 항목을 찾는다. 못 찾으면 None."""
    q = _norm(name)
    for entry in TECHNIQUES.values():
        if q in [_norm(a) for a in entry["aliases"]] or q == _norm(entry["en"]) \
                or q == _norm(entry["ko"]):
            return entry
        # 부분 포함(예: "프렌치 드롭으로" → "프렌치 드롭")
        for a in entry["aliases"]:
            if _norm(a) in q or q in _norm(a):
                return entry
    return None


def entry_to_dict(entry: dict) -> dict:
    """출력/JSON용 표현 (참고 영상 링크 포함)."""
    return {
        "name_ko": entry["ko"], "name_en": entry["en"], "type": entry["type"],
        "desc": entry["desc"], "cues": entry["cues"],
        "reference_url": _yt(entry["query"]),
    }


def search_url(name: str) -> str:
    """용어집에 없는 기법명에 대한 YouTube 검색 링크."""
    return _yt(f"{name} magic sleight tutorial")
