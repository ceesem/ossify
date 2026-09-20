"""Execute the code blocks in the user guide so the examples cannot rot.

Doc examples are usually the first code a new user runs, and nothing else in
the suite touches them -- so an API change can silently invalidate a whole
page. These tests run each page's ``python`` blocks the way a reader would:
top to bottom, in one shared namespace, where a name defined in an early block
is still around later.

Two accommodations make that practical:

* Pages load the example neuron from a GitHub URL, which is the right thing to
  show a reader. The same file is committed at the repo root, so the loader is
  redirected to the local copy and the tests stay offline and fast.
* Blocks run with the working directory set to a temporary folder, because some
  pages write files. Without this, running the guide litters the repo.

Only pages listed in ``EXECUTABLE_PAGES`` are covered. The rest of the guide
has known-failing examples; add a page here once it has been fixed, and this
test will keep it honest.
"""

import io
import os
import re
import textwrap
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import ossify  # noqa: E402  (after the Agg backend is selected)

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# The example neuron the guide downloads, and the copy committed alongside it.
EXAMPLE_URL = (
    "https://github.com/ceesem/ossify/raw/refs/heads/main/864691135336055529.osy"
)
EXAMPLE_LOCAL = REPO / "864691135336055529.osy"

# Pages whose examples are known to run end to end.
#
# Three guide pages are deliberately absent, because their examples cannot be
# executed rather than because they are broken. Their API usage has been
# checked by running them against a pre-seeded namespace:
#
# * faq.md -- standalone fragments that answer one question each, written
#   against placeholder names a reader supplies.
# * visualization_and_plotting.md -- the later sections need PyVista (the
#   optional `viz` extra) and a rendering context.
# * data_import_export.md -- reads and writes against CAVE, cloud storage and
#   file paths the reader provides.
EXECUTABLE_PAGES = [
    "algorithms_and_analysis.md",
    "cell_object.md",
    "getting_started.md",
    "linking_and_mapping.md",
    "masking_and_filtering.md",
    "shared_layer_features.md",
    "working_with_annotations.md",
    "working_with_graphs.md",
    "working_with_meshes.md",
    "working_with_skeletons.md",
]


def _python_blocks(text: str):
    """Yield ``(source, first_line_number)`` for each python fence.

    Fences nested inside a list item are indented, which is valid Markdown but
    not valid Python, so the source is dedented before it is handed back.
    """
    for match in re.finditer(r"^([ \t]*)```python\n(.*?)^\1```", text, re.S | re.M):
        yield textwrap.dedent(match.group(2)), text[: match.start()].count("\n") + 2


@pytest.fixture
def local_example_cell(monkeypatch):
    """Serve the documented example URL from the committed copy."""
    if not EXAMPLE_LOCAL.exists():
        pytest.skip(f"{EXAMPLE_LOCAL.name} is not present")
    real_load = ossify.load_cell

    def load(path, *args, **kwargs):
        if isinstance(path, str) and path == EXAMPLE_URL:
            path = str(EXAMPLE_LOCAL)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(ossify, "load_cell", load)
    monkeypatch.setattr(ossify.file_io, "load_cell", load)
    return load


@pytest.mark.parametrize("page", EXECUTABLE_PAGES)
def test_doc_page_examples_run(page, local_example_cell, tmp_path, monkeypatch):
    path = DOCS / page
    assert path.exists(), f"{page} is listed as executable but does not exist"
    blocks = list(_python_blocks(path.read_text()))
    assert blocks, f"no python blocks found in {page}"

    # Some pages write files; keep them out of the repo.
    monkeypatch.chdir(tmp_path)

    namespace: dict = {"ossify": ossify}
    for index, (source, line_no) in enumerate(blocks):
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


def test_example_cell_is_committed():
    """The redirect above is what keeps these tests offline; if the file goes
    missing every page test silently skips instead of failing."""
    assert EXAMPLE_LOCAL.exists(), (
        f"{EXAMPLE_LOCAL.name} is referenced by the guide and needed by the doc "
        "tests; it must stay in the repository root."
    )


def test_doc_runs_do_not_touch_the_repo(tmp_path, monkeypatch):
    """Guard the guard: the chdir above must actually take effect."""
    monkeypatch.chdir(tmp_path)
    assert Path(os.getcwd()).resolve() == tmp_path.resolve()
    assert REPO not in Path(os.getcwd()).resolve().parents
