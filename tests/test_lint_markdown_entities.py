"""§18a — HTML entities stranded inside Markdown code spans.

Markdown decodes nothing between backticks, so ``a&#95;b`` renders as seven
literal characters rather than ``a_b``. The entity form is correct in a
**Mermaid** node label (§14, §18), where Mermaid decodes it after its lexer
has run, and wrong everywhere else -- inside a code span the character is
already literal.

This was a real, twice-repeated failure: a hand-written table rendered as
``WSD&#95;STABLE&#95;FRAC`` across ten rows on 2026-09-24, and eleven more
instances appeared in a second document the next day. Both came from copying
the Mermaid rule into prose. The rule exists so the third time is caught.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from lint_markdown import RULES, lint  # noqa: E402


def _lint(tmp_path, body: str, name: str = "doc.md"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return lint(p)


def _ids(findings):
    return [f.rule for f in findings]


def test_rule_is_registered():
    assert "§18a" in RULES


def test_entity_inside_code_span_is_fatal(tmp_path):
    found = _lint(tmp_path, "The knob `WSD&#95;STABLE&#95;FRAC` is 0.60.\n")
    assert _ids(found) == ["§18a"]
    assert found[0].severity == "FATAL"
    assert found[0].line == 1


def test_plain_underscore_in_code_span_is_clean(tmp_path):
    """The fix: inside backticks the underscore needs no escaping."""
    assert _lint(tmp_path, "The knob `WSD_STABLE_FRAC` is 0.60.\n") == []


def test_entity_in_unbackticked_prose_is_allowed(tmp_path):
    """Outside code spans the entity is the documented workaround for the
    Markdown emphasis pass, so it must not fire."""
    assert _lint(tmp_path, "The V&#95;theta bank is shared across layers.\n") == []


def test_entity_in_mermaid_label_is_allowed(tmp_path):
    """§18 *requires* the entity here; §18a must not contradict it."""
    body = '```mermaid\nflowchart LR\n  A["h&#95;t"] --> B["r&#95;new"]\n```\n'
    assert "§18a" not in _ids(_lint(tmp_path, body))


def test_fenced_code_block_is_exempt(tmp_path):
    """A fenced block may legitimately *quote* the broken form -- the
    cheatsheet itself does."""
    body = "```text\nA[\"h&#95;t\"]\n```\n"
    assert "§18a" not in _ids(_lint(tmp_path, body))


def test_named_entity_is_caught(tmp_path):
    found = _lint(tmp_path, "Write `a &amp; b` in the cell.\n")
    assert _ids(found) == ["§18a"]


@pytest.mark.parametrize("line,expect", [
    ("Set `x&#95;y` here.", 1),                      # one span, one entity
    ("Both `a&#95;b` and `c&#95;d` broke.", 2),      # two spans
    ("Only `a_b` is fine, `c&#95;d` is not.", 1),    # mixed
])
def test_counts_one_finding_per_span(tmp_path, line, expect):
    assert len(_lint(tmp_path, line + "\n")) == expect


def test_line_number_survives_a_preceding_fence(tmp_path):
    """Segment offsets are easy to get wrong; pin them."""
    body = "intro\n\n```python\nx = 1\n```\n\nThe knob `A&#95;B` here.\n"
    found = _lint(tmp_path, body)
    assert _ids(found) == ["§18a"]
    assert found[0].line == 7, found[0].line


# ---------------------------------------------------------------------------
# Exemptions: code spans that legitimately *show* markup
# ---------------------------------------------------------------------------

def test_double_backticks_quote_markup_verbatim(tmp_path):
    """``` ``...`` ``` is how you quote a single-backtick construct."""
    assert _lint(tmp_path, "Write `` `A&#95;B` `` to show the broken form.\n") == []


def test_span_that_is_only_the_entity_names_it(tmp_path):
    """A document explaining the rule must be able to name `&#95;`."""
    assert _lint(tmp_path, "Reach for `&#95;` only in a Mermaid label.\n") == []


def test_span_holding_mermaid_node_syntax_is_exempt(tmp_path):
    """Inside `["..."]` the entity is *required* by §14/§18."""
    assert _lint(tmp_path, 'Use `A["h&#95;t"]` in the diagram.\n') == []


# ---------------------------------------------------------------------------
# File-level suppression
# ---------------------------------------------------------------------------

def test_lint_disable_suppresses_the_rule(tmp_path):
    body = "<!-- lint-disable §18a -->\n\nThe knob `A&#95;B` here.\n"
    assert _lint(tmp_path, body) == []


def test_lint_disable_is_rule_specific(tmp_path):
    """Disabling one rule must not silence the others."""
    body = "<!-- lint-disable §13 -->\n\nThe knob `A&#95;B` here.\n"
    assert _ids(_lint(tmp_path, body)) == ["§18a"]


def test_lint_disable_accepts_several_rules(tmp_path):
    body = "<!-- lint-disable §13, §18a -->\n\nThe knob `A&#95;B` here.\n"
    assert _lint(tmp_path, body) == []
