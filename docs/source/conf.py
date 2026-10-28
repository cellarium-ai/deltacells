# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Sphinx configuration for the deltacells documentation."""

import importlib.metadata

project = "deltacells"
author = "Stephen Fleming and Cellarium contributors"
copyright = "Contributors to the Cellarium project"
try:
    release = importlib.metadata.version("deltacells")
except importlib.metadata.PackageNotFoundError:
    release = "dev"
version = release

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_title = "deltacells"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/cellarium-ai/deltacells",
    "repository_branch": "main",
    "path_to_docs": "docs/source",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
}
