import re
from pathlib import Path

README = (Path(__file__).parents[1] / "README.md").read_text()


def test_every_link_and_picture_works_on_pypi_too():
    # PyPI shows this README but can't resolve paths in the repository or jump to headings: every address is a full URL
    targets = re.findall(r"\]\(([^)\s]+)\)", README) + re.findall(r'(?:href|src|srcset)="([^"]+)"', README)
    relative = [t for t in targets if not t.startswith(("https://", "mailto:"))]
    assert targets and not relative, f"use full https:// addresses in README.md: {relative}"
