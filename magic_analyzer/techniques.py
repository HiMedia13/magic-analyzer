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
        "aliases": ["classic palm", "클래식 팜", "팜", "palm", "palming",
                    "팔밍", "팔름", "파밍", "팜링"],
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
    "color_change": {
        "ko": "컬러 체인지", "en": "Color Change",
        "aliases": ["color change", "컬러 체인지", "컬러체인지", "colour change", "색 변화"],
        "type": "card",
        "desc": ("보이던 카드가 순식간에 다른 카드로 '변하는' 시각적 효과. 손이나 다른 카드가 "
                 "표면을 스치고 지나가는 순간, 앞면 카드를 비밀리에 바꿔치기한다."),
        "cues": "손이 카드 앞면을 쓸고 지나간 직후 다른 카드, 빠른 손동작(FAST)",
        "query": "card color change tutorial slow motion",
    },
    "elmsley_count": {
        "ko": "엘름슬리 카운트", "en": "Elmsley Count",
        "aliases": ["elmsley count", "elmsley", "엘름슬리", "엘름슬리 카운트", "ghost count"],
        "type": "card",
        "desc": ("4장을 4장처럼 세는 척하지만 그중 한 장을 숨겨, 실제와 다른 카드 구성을 "
                 "보여준다. 패킷(소수 카드) 트릭의 기본기. 카운트 중 한 장이 슬쩍 감춰진다."),
        "cues": "카드를 한 장씩 세는 동작 중 한 장이 가려짐, 손가락 미세 조작",
        "query": "elmsley count tutorial",
    },
    "false_shuffle": {
        "ko": "가짜 셔플", "en": "False Shuffle",
        "aliases": ["false shuffle", "가짜 셔플", "거짓 셔플", "hindu shuffle control",
                    "false riffle shuffle", "셔플 컨트롤"],
        "type": "card",
        "desc": ("섞는 것처럼 보이지만 특정 카드(보통 맨 위/아래)나 전체 순서를 그대로 "
                 "유지한다. 힌두 셔플로 바텀 카드를 지키는 컨트롤이 대표적."),
        "cues": "셔플하는데도 결정적 카드가 같은 위치에 유지됨",
        "query": "false shuffle card control tutorial",
    },
    "card_force": {
        "ko": "포스", "en": "Force",
        "aliases": ["force", "포스", "card force", "카드 포스", "riffle force", "classic force"],
        "type": "card",
        "desc": ("관객이 '자유롭게' 고른 것처럼 느끼지만, 실제로는 마술사가 미리 정한 카드를 "
                 "집게 만든다. 리플 포스(멈추라고 한 지점에 그 카드를 두는 식), 클래식 포스 등."),
        "cues": "선택 타이밍을 마술사가 유도, 덱을 넘기다 특정 순간에 멈추게 함",
        "query": "card force riffle classic tutorial",
    },
    "glide": {
        "ko": "글라이드", "en": "Glide",
        "aliases": ["glide", "글라이드"],
        "type": "card",
        "desc": ("덱 맨 아래 카드를 빼는 척하면서, 손가락으로 그 카드를 살짝 당겨두고 "
                 "대신 두 번째 아래 카드를 빼낸다. 관객은 바닥 카드가 나왔다고 믿는다."),
        "cues": "바닥에서 카드를 빼는 순간 손가락이 바닥 카드를 살짝 뒤로 당김",
        "query": "glide card sleight tutorial",
    },
    "card_palm": {
        "ko": "카드 팜(탑 팜)", "en": "Card Palm",
        "aliases": ["card palm", "탑 팜", "top palm", "카드 팜", "palm card"],
        "type": "card",
        "desc": ("덱 맨 위 카드(들)를 손바닥에 몰래 숨겨 빼낸다. 손은 자연스럽게 비어 보이지만 "
                 "손바닥 안에 카드가 있다. 카드를 주머니/다른 곳으로 옮기는 데 쓴다."),
        "cues": "덱을 덮은 손이 떨어질 때 손바닥이 미묘하게 굽음, 손바닥 노출 회피",
        "query": "top palm card tutorial",
    },
    "second_deal": {
        "ko": "세컨드 딜", "en": "Second Deal",
        "aliases": ["second deal", "세컨드 딜", "세컨딜", "second dealing"],
        "type": "card",
        "desc": ("맨 위 카드를 미는 척하면서 실제로는 바로 아래(두 번째) 카드를 딜한다. "
                 "맨 위 카드를 보존하는 도박/마술 기술. 고난도."),
        "cues": "딜할 때 맨 위 카드가 살짝 밀렸다 되돌아오고 둘째 카드가 나감",
        "query": "second deal tutorial slow motion",
    },
    "bobo_switch": {
        "ko": "보보 스위치", "en": "Bobo Switch",
        "aliases": ["bobo switch", "보보 스위치", "보보스위치", "bobo"],
        "type": "coin",
        "desc": ("동전을 한 손에서 다른 손으로 옮기는 '척'하면서 실제로는 핑거팜으로 원래 손에 "
                 "남긴다. 코인 어크로스/매트릭스 등에서 개수를 속이는 표준 스위치."),
        "cues": "건네는 제스처(CONTACT) 직후 받은 손이 빈손, 원손 손가락 말림",
        "query": "bobo switch coin tutorial",
    },
    "han_ping_chien": {
        "ko": "한 핑 첸", "en": "Han Ping Chien",
        "aliases": ["han ping chien", "한 핑 첸", "한핑첸", "hpc"],
        "type": "coin",
        "desc": ("한 손의 동전들을 테이블/다른 손으로 옮기는 큰 동작 속에서, 다른 손에 숨겨둔 "
                 "동전을 비밀리에 떨어뜨려 개수가 순간 이동한 것처럼 보이게 한다."),
        "cues": "양손이 동시에 움직이는 큰 제스처(FAST), 한쪽에서 몰래 릴리스",
        "query": "han ping chien coin move tutorial",
    },
    "muscle_pass": {
        "ko": "머슬 패스", "en": "Muscle Pass",
        "aliases": ["muscle pass", "머슬 패스", "머슬패스", "근육 패스"],
        "type": "coin",
        "desc": ("손가락을 쓰지 않고 손바닥 근육의 압력만으로 동전을 한 손에서 다른 손으로 "
                 "'튕겨' 보낸다. 순간이동처럼 보이는 고난도 플러리시 겸 비밀 이동."),
        "cues": "손가락 움직임 없이 동전이 손 사이를 순간 이동",
        "query": "muscle pass coin tutorial",
    },
    "spellbound": {
        "ko": "스펠바운드", "en": "Spellbound",
        "aliases": ["spellbound", "스펠바운드"],
        "type": "coin",
        "desc": ("손가락 사이에 든 동전이 계속 다른 동전(은화↔동화 등)으로 바뀌는 연속 변화. "
                 "두 동전을 썸팜/핀치로 번갈아 보여주며 하나처럼 속인다."),
        "cues": "엄지로 쓸어내릴 때마다 앞 동전이 바뀜, 손등 뒤에 둘째 동전 은닉",
        "query": "spellbound coin routine tutorial",
    },
    "lapping": {
        "ko": "래핑", "en": "Lapping",
        "aliases": ["lapping", "래핑", "랩핑", "lap"],
        "type": "coin/card 겸용",
        "desc": ("물체를 몰래 무릎(lap) 위로 떨어뜨려 손에서 사라진 것처럼 만든다. 앉아서 하는 "
                 "테이블 마술에서 흔하며, 손은 아무것도 안 한 듯 깨끗해 보인다."),
        "cues": "물체를 쥔 손이 테이블 가장자리로 가는 순간 사라짐(BORDER/VANISH)",
        "query": "lapping magic technique tutorial",
    },
    "misdirection": {
        "ko": "미스디렉션", "en": "Misdirection",
        "aliases": ["misdirection", "미스디렉션", "미스디렉션", "시선 분산", "주의 분산"],
        "type": "심리(공통)",
        "desc": ("손기술 자체가 아니라, 결정적 비밀 동작의 순간에 관객의 시선·주의를 다른 곳으로 "
                 "돌리는 심리 기법. 시선·말·큰 동작으로 작은 비밀 동작을 가린다. 모든 슬레이트의 "
                 "성공을 좌우한다."),
        "cues": "비밀 동작 순간에 반대 손/얼굴/말로 주의를 끎",
        "query": "misdirection magic explained",
    },
    "gimmick": {
        "ko": "기믹(특수 도구)", "en": "Gimmick",
        "aliases": ["gimmick", "기믹", "기믹 코인", "gimmicked", "장치"],
        "type": "도구(공통)",
        "desc": ("손기술이 아니라 특수 제작된 도구로 효과를 낸다 — 기믹 코인(접히는/껍데기 동전), "
                 "더블페이스 카드, 자석, 투명실(IT) 등. 영상에서 손동작이 너무 깨끗한데 효과가 "
                 "강하면 기믹을 의심할 수 있다."),
        "cues": "뚜렷한 슬레이트 없이 효과 발생, 특정 각도만 보여줌",
        "query": "magic gimmick coin card explained",
    },
    "cull": {
        "ko": "컬(스프레드 컬)", "en": "Cull",
        "aliases": ["cull", "컬", "spread cull", "스프레드 컬", "hofzinser cull"],
        "type": "card",
        "desc": ("카드를 펼치거나 넘기는 동안 특정 카드(들)를 비밀리에 골라 덱 아래나 원하는 "
                 "위치로 모은다. 관객에게 보여주는 척하면서 목표 카드를 컨트롤한다."),
        "cues": "카드를 스프레드/넘기는 중 손가락이 특정 카드를 아래로 끌어모음",
        "query": "spread cull card control tutorial",
    },
    "false_cut": {
        "ko": "가짜 컷", "en": "False Cut",
        "aliases": ["false cut", "가짜 컷", "거짓 컷", "zarrow", "charlier cut", "swing cut false"],
        "type": "card",
        "desc": ("덱을 자르는 것처럼 보이지만 전체(또는 핵심 카드) 순서를 그대로 유지한다. "
                 "셋업을 지키면서 섞은 듯한 인상을 준다. 자로우/차알리어 등 변형이 많다."),
        "cues": "여러 번 자르는 듯 보여도 결정적 순서가 보존됨",
        "query": "false cut card tutorial",
    },
    "pinky_break": {
        "ko": "핑키 브레이크", "en": "Pinky Break",
        "aliases": ["pinky break", "핑키 브레이크", "브레이크", "break", "pinky count"],
        "type": "card",
        "desc": ("새끼손가락으로 덱 사이에 미세한 틈(브레이크)을 잡아 특정 카드의 위치를 비밀리에 "
                 "표시·유지한다. 더블리프트·패스·컨트롤의 사전 동작(겟 레디)으로 널리 쓰인다."),
        "cues": "덱을 쥔 새끼손가락이 살짝 안쪽으로, 카드 사이 미세 틈 유지",
        "query": "pinky break get ready card tutorial",
    },
    "bottom_deal": {
        "ko": "바텀 딜", "en": "Bottom Deal",
        "aliases": ["bottom deal", "바텀 딜", "바텀딜", "bottom dealing"],
        "type": "card",
        "desc": ("맨 위 카드를 딜하는 척하면서 실제로는 덱 맨 아래 카드를 딜한다. 도박/마술의 "
                 "고난도 기술로, 맨 위 카드를 보존하거나 원하는 카드를 분배할 때 쓴다."),
        "cues": "딜하는 순간 덱 아래쪽에서 카드가 빠져나옴, 그립 변화",
        "query": "bottom deal card tutorial slow motion",
    },
    "side_steal": {
        "ko": "사이드 스틸", "en": "Side Steal",
        "aliases": ["side steal", "사이드 스틸", "사이드스틸"],
        "type": "card",
        "desc": ("덱 중간에 있는 카드를 옆으로 비밀리에 빼내 팜하거나 맨 위로 옮기는 컨트롤. "
                 "관객이 끼운 카드를 몰래 가져오는 데 쓴다."),
        "cues": "덱을 쥔 손에서 카드 한 장이 옆으로 빠져 손바닥으로",
        "query": "side steal card sleight tutorial",
    },
    "downs_palm": {
        "ko": "다운스 팜", "en": "Downs Palm",
        "aliases": ["downs palm", "다운스 팜", "다운스팜", "t nelson downs palm"],
        "type": "coin",
        "desc": ("T. Nelson Downs가 정립한 동전 팜 — 엄지 밑동(웹)과 손바닥 근육으로 동전을 "
                 "물어, 손가락을 펴 보여도 동전이 가려진다. 동전 프로덕션·베니시의 고전."),
        "cues": "손을 펴 보여도 엄지 밑동에 동전 은닉, 손등 회전 회피",
        "query": "downs palm coin tutorial",
    },
    "back_palm": {
        "ko": "백 팜", "en": "Back Palm",
        "aliases": ["back palm", "백 팜", "백팜", "back palming"],
        "type": "coin/card 겸용",
        "desc": ("동전/카드를 손등 쪽으로 숨겨, 손바닥을 펴서 정면으로 보여줘도 비어 보이게 한다. "
                 "매니퓰레이션(카드/동전 프로덕션)의 핵심 — 손을 뒤집으며 앞뒤로 숨긴다."),
        "cues": "손바닥을 보여줄 때 물체가 손등 뒤로, 손 뒤집기 동작",
        "query": "back palm coin card manipulation tutorial",
    },
    "edge_grip": {
        "ko": "엣지 그립(텐카이)", "en": "Edge Grip",
        "aliases": ["edge grip", "엣지 그립", "tenkai", "텐카이", "tenkai palm"],
        "type": "coin",
        "desc": ("엄지 밑동으로 동전의 '가장자리'를 물어 쥐어, 손가락을 펴거나 손을 기울여도 "
                 "동전이 가려진다(텐카이 팜). 손이 비어 보이는 베니시에 쓴다."),
        "cues": "동전이 엄지 안쪽 가장자리에 세워져 은닉, 손바닥 정면 회피",
        "query": "tenkai palm edge grip coin tutorial",
    },
    "sleeving": {
        "ko": "슬리빙", "en": "Sleeving",
        "aliases": ["sleeving", "슬리빙", "슬리브", "sleeve"],
        "type": "coin/card 겸용",
        "desc": ("물체를 소매 안으로 몰래 떨어뜨려/밀어넣어 사라지게 한다('소매 속으로'). "
                 "손은 깨끗해 보이고 물체가 순간 사라진 듯하다."),
        "cues": "손이 소매 근처를 스치는 순간 물체 사라짐(VANISH)",
        "query": "sleeving magic coin technique tutorial",
    },
    "topit": {
        "ko": "토핏", "en": "Topit",
        "aliases": ["topit", "토핏", "탑잇"],
        "type": "coin/card 겸용",
        "desc": ("재킷 안쪽에 단 비밀 주머니(토핏)에 물체를 떨어뜨려 사라지게 하는 도구+기법. "
                 "손을 몸 쪽으로 가져가는 자연스러운 동작에 물체를 흘려 넣는다."),
        "cues": "쥔 손이 가슴/재킷 앞으로 가는 순간 물체 사라짐",
        "query": "topit magic vanish tutorial",
    },
}


def _norm(s: str) -> str:
    return s.strip().lower().replace("-", " ")


def lookup(name: str) -> dict | None:
    """기법명(한/영, 느슨)으로 용어집 항목을 찾는다. 못 찾으면 None.

    매칭 우선순위: (1) 정확히 일치하면 즉시 반환, (2) 별칭이 질의 안에 포함되면
    '가장 긴 별칭'이 매칭된 항목을 택한다. (예: '머슬 패스'가 '패스'보다 우선)
    """
    q = _norm(name)
    best, best_len = None, 0
    for entry in TECHNIQUES.values():
        cands = [_norm(a) for a in entry["aliases"]]
        cands += [_norm(entry["en"]), _norm(entry["ko"])]
        for a in cands:
            if not a:
                continue
            if q == a:
                return entry              # 정확 일치 = 최우선
            if a in q and len(a) > best_len:  # 별칭이 질의에 포함 → 가장 긴 것 채택
                best, best_len = entry, len(a)
    return best


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
