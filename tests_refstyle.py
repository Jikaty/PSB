import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import config
import prompts
from config import FLAGS
from eval_run import facts, ref_cover, select_cases

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL", name)


TODAY = "15.11.2026."

check("flag default off", config.FLAG_DEFAULTS["USE_REF_STYLE"] is False)
check("ref rules: base text present once", all(prompts.SYSTEM_PROMPT.count(old) == 1 for old, _ in prompts.REF_RULES))

base = prompts.build_system_prompt(TODAY, False, True, False, False)
ref = prompts.build_system_prompt(TODAY, False, True, False, True)
check("off: prompt unchanged", base == prompts.SYSTEM_PROMPT.format(today=TODAY))
check("on: new rules in", all(new in ref for _, new in prompts.REF_RULES))
check("on: old rules out", all(old not in ref for old, _ in prompts.REF_RULES))
check("on: 30–80 words", "30–80 слов" in ref and "1–4 предложения" not in ref)
check("on: format block kept", "ФОРМАТ ВЫВОДА" in ref and '"citations"' in ref)
check("on: format braces rendered", "{{" not in ref and "{today}" not in ref)

both = prompts.build_system_prompt(TODAY, False, True, True, True)
check("ref wins over full", all(new in both for _, new in prompts.REF_RULES) and not any(new in both for _, new in prompts.FULL_RULES))
check("full alone unchanged", all(new in prompts.build_system_prompt(TODAY, False, True, True, False) for _, new in prompts.FULL_RULES))
check("ref with step7 off", all(new in prompts.build_system_prompt(TODAY, False, False, False, True) for _, new in prompts.REF_RULES))
check("ref with extra rules", prompts.EXTRA_RULES in prompts.build_system_prompt(TODAY, True, True, False, True))

FLAGS["USE_REF_STYLE"] = True
check("flag drives prompt", prompts.build_system_prompt(TODAY) == prompts.build_system_prompt(TODAY, ref=True))
ver = prompts.build_verifier_system(TODAY)
check("verifier: ref length", "Верные дополнительные положения" in ver and prompts.VERIFIER_LENGTH not in ver)
FLAGS["USE_REF_STYLE"] = False
ver_base = prompts.build_verifier_system(TODAY)
check("verifier: base unchanged", prompts.VERIFIER_LENGTH in ver_base and "Верные дополнительные" not in ver_base)
check("verifier: full unchanged", prompts.VERIFIER_FULL_LENGTH in prompts.build_verifier_system(TODAY, full=True))
check("verifier: ref wins over full", "Верные дополнительные" in prompts.build_verifier_system(TODAY, full=True, ref=True))

g1 = "Арендная плата по договору № А-08/2024 — 450 000 рублей (п. 3.1). Не позднее 5-го числа (п. 3.2). Коммунальные отдельно (п. 3.3)."
check("facts: locators", {"п3.1", "п3.2", "п3.3"} <= facts(g1))
check("facts: doc number", "А-08/2024" in facts(g1))
check("facts: grouped number", "450000" in facts(g1) and "450" not in facts(g1) and "000" not in facts(g1))
check("facts: locator digits not numbers", "3.1" not in facts(g1))
check("facts: dates and laws", {"10.01.2023", "152-ФЗ", "ст9"} <= facts("ред. от 10.01.2023, 152-ФЗ, ст. 9"))
check("facts: nbsp grouping", "10000000" in facts("10 000 000 рублей"))
check("cover: self", ref_cover(g1, g1) == 1.0)
check("cover: partial", abs(ref_cover(g1, "450 000 рублей, до 5-го числа, п. 3.1 и п. 3.2, договор А-08/2024") - 5 / 6) < 1e-9)
check("cover: no facts", ref_cover("срок не более шести месяцев", "шесть месяцев") is None)
check("cover: empty answer", ref_cover(g1, "") == 0.0)
check("cover: statute words", ref_cover("ТК РФ, ст. 70", "ТК РФ, статья 70") == 1.0)
check("cover: clause words", ref_cover("п. 4.2 и ч. 2", "пункт 4.2, часть 2") == 1.0)
check("facts: plain word not locator", facts("стоимость 5 рублей") == {"5"})

cases = [{"id": "1", "category": "direct", "question": "q"}, {"id": "j01", "category": "injection", "question": "q"}]
check("demo selects numeric ids", [c["id"] for c in select_cases(cases, "", "", True)] == ["1"])
check("demo off keeps all", len(select_cases(cases, "", "", False)) == 2)
check("demo with ids", select_cases(cases, "", "j01", True) == [])

print(f"passed {passed}, failed {failed}")
sys.exit(1 if failed else 0)
