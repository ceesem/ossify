"""Execute the code blocks in the user guide so the examples cannot rot.

Doc examples are usually the first code a new user runs, and nothing else in
the suite touches them -- so an API change can silently invalidate a whole
page. These tests run each page's ``python`` blocks the way a reader would:
top to bottom, in one shared namespace, where a name defined in an early block
is still around later.

Only pages listed in ``EXECUTABLE_PAGES`` are covered. The rest of the guide
has known-failing examples; add a page here once it has been fixed, and this
test will keep it honest.
"""

import io
import re
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Pages whose examples are known to run end to end.
EXECUTABLE_PAGES = ["masking_and_filtering.md"]

# Blocks that reach the network are shown but not executed here.
NETWORK_MARKERS = ("http://", "https://")


def _python_blocks(text: str):
    """Yield ``(source, first_line_number)`` for each python fence."""
    for match in re.finditer(r"```python\n(.*?)```", text, re.S):
        yield match.group(1), text[: match.start()].count("\n") + 2


@pytest.mark.parametrize("page", EXECUTABLE_PAGES)
def test_doc_page_examples_run(page):
    path = DOCS / page
    assert path.exists(), f"{page} is listed as executable but does not exist"
    blocks = list(_python_blocks(path.read_text()))
    assert blocks, f"no python blocks found in {page}"

    namespace: dict = {}
    executed = 0
    for index, (source, line_no) in enumerate(blocks):
        if any(marker in source for marker in NETWORK_MARKERS):
            continue
        filename = f"{page}:block{index}"
        try:
            # Doc examples print liberally; keep the test output readable.
            with redirect_stdout(io.StringIO()):
                exec(compile(source, filename, "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - re-raised with better context
            frames = traceback.extract_tb(exc.__traceback__)
            offset = next((f.lineno for f in frames if f.filename == filename), None)
            failing = source.splitlines()[offset - 1].strip() if offset else "<unknown>"
            pytest.fail(
                f"{page} example failed at line ~{line_no + (offset or 1) - 1}: "
                f"{failing}\n    {type(exc).__name__}: {exc}",
                pytrace=False,
            )
        executed += 1
    assert executed > 0, f"every block in {page} was skipped"


def test_network_blocks_are_skipped_not_silently_passing():
    """The skip rule must not quietly swallow the whole page."""
    text = (DOCS / "masking_and_filtering.md").read_text()
    blocks = list(_python_blocks(text))
    skipped = [b for b, _ in blocks if any(m in b for m in NETWORK_MARKERS)]
    assert len(skipped) < len(blocks), "all blocks were treated as network blocks"
