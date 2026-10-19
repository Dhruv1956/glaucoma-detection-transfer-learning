"""Tiny helper that prints clear PASS / FAIL lines and counts failures."""

_failures = []


def check(name, condition, detail=""):
    """Print PASS or FAIL for one sanity check. Returns True when it passed."""
    ok = bool(condition)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)
    return ok


def summary():
    """Print how many checks failed so far (call at the end of a script)."""
    if _failures:
        print(f"\n*** {len(_failures)} CHECK(S) FAILED: {_failures}")
    else:
        print("\nAll checks passed.")
    return len(_failures) == 0
