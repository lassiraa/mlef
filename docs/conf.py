"""Build the public documentation with Sphinx's bundled theme and search."""

from importlib.metadata import version as _version

project = "MLEF"
author = "MLEF contributors"
release = _version("mlef")
version = release
extensions = ["sphinx.ext.githubpages"]
root_doc = "index"
exclude_patterns = []
html_theme = "alabaster"
html_title = f"{project} documentation"
html_baseurl = "https://lassiraa.github.io/mlef/"
html_theme_options = {
    "github_user": "lassiraa",
    "github_repo": "mlef",
    "github_banner": True,
    "description": "Reproducible Python experiments and tabular model feedback",
}
html_sidebars = {"**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]}
