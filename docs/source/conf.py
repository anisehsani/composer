# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib
import json
import os
import sys
import textwrap
import types
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch.nn

import sphinx.application
import sphinx.ext.autodoc
import sphinx.util.logging
from sphinx.ext.autodoc import ClassDocumenter, _
import yahp as hp

sys.path.insert(0, os.path.abspath('..'))

log = sphinx.util.logging.getLogger(__name__)

# -- Project information -----------------------------------------------------

project = 'MosaicML'
copyright = '2022, MosaicML, Inc.'
author = 'MosaicML'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    'sphinxcontrib.katex',
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    'sphinxemoji.sphinxemoji',
    "sphinxext.opengraph",
    "sphinx_copybutton",
    "myst_parser",
    "sphinxarg.ext",
    'sphinx.ext.doctest',
    'sphinx_panels',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tables/algorithms_table.md']

napoleon_custom_sections = [('Returns', 'params_style')]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Make sure the target is unique
autosectionlabel_prefix_document = True
autosummary_imported_members = False
autosectionlabel_maxdepth = 5
autosummary_generate = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_title = " "

# Customize CSS
html_css_files = ['css/custom.css']
html_js_files = [
    'js/posthog.js',
]

# Mosaic logo
# html_logo = 'https://storage.googleapis.com/docs.mosaicml.com/images/logo-dark-bg.png'
html_theme_options = {
    "light_logo": "logo-light-mode.png",
    "dark_logo": "logo-dark-mode.png",
    "light_css_variables": {
        "color-brand-primary": "#343434",
        "color-brand-content": "#343434",
    },
    "dark_css_variables": {
        "color-brand-primary": "#f9f9f9",
        "color-brand-content": "#f9f9f9",
    },
}

# Favicon
html_favicon = 'https://mosaic-ml-staging.cdn.prismic.io/mosaic-ml-staging/b1f1a2a0-2b54-4b43-9b76-bfa2e24d6fdf_favicon.svg'

# Don't unfold our common type aliases
autodoc_type_aliases = {
    'Tensor': 'composer.core.types.Tensor',
    'Tensors': 'composer.core.types.Tensors',
    'Batch': 'composer.core.types.Batch',
    'BatchPair': 'composer.core.types.BatchPair',
    'BatchDict': 'composer.core.types.BatchDict',
    'StateDict': 'composer.core.types.StateDict',
    'TDeviceTransformFn': 'composer.core.types.TDeviceTransformFn',
    'Hparams': 'yahp.hparams.Hparams',
}

autodoc_default_options = {
    # don't document the forward() method. Because of how torch.nn.Module.forward is defined in the
    # base class, sphinx does not realize that forward overrides an inherited method.
    'exclude-members': 'hparams_registry'
}
autodoc_inherit_docstrings = False

# Monkeypatch some objects as to exclude their docstrings
hp.Hparams.__doc__ = ""
hp.Hparams.initialize_object.__doc__ = ""
torch.nn.Module.forward.__doc__ = ""

pygments_style = "manni"
pygments_dark_style = "monokai"

html_permalinks = True
html_permalinks_icon = "#"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'yapf': ('https://docs.mosaicml.com/projects/yahp/en/stable/', None),
    'torchvision': ('https://pytorch.org/vision/stable/', None),
    'torchtext': ('https://pytorch.org/text/stable/', None),
    'torchmetrics': ('https://torchmetrics.readthedocs.io/en/latest/', None),
    'libcloud': ('https://libcloud.readthedocs.io/en/stable/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable', None),
}

nitpicky = False  # warn on broken links

nitpick_ignore = [
    ('py:class', 'type'),
    ('py:class', 'optional'),
    ('py:meth', 'wandb.init'),
    ('py:attr', 'wandb.run.config'),
    ('py:attr', 'wandb.run.tags'),
    ('py:meth', 'torch.save'),
    ('py:meth', 'torch.load'),
    ('py:class', 'TLogDataValue'),
    ('py:class', 'TLogData'),
    ('py:class', 'T_nnModule'),
]

python_use_unqualified_type_names = True
autodoc_typehints = "none"


def skip_redundant_namedtuple_attributes(
    app: sphinx.application.Sphinx,
    what: str,
    name: str,
    obj: Any,
    skip: bool,
    options: sphinx.ext.autodoc.Options,
):
    # Hide the default, duplicate attributes for named tuples
    del app, what, name, skip, options
    if '_tuplegetter' in obj.__class__.__name__:
        return True
    return None


with open(os.path.join(os.path.dirname(__file__), "doctest_fixtures.py"), "r") as f:
    doctest_global_setup = f.read()


def determine_sphinx_path(item: Union[Type[object], Type[BaseException], types.MethodType, types.FunctionType],
                          module_name: str) -> Optional[str]:
    """Returns the path to where an item is documented.

    #. If ``item`` is private, then a Sphinx warning is emitted, as private members should not be documented
    #. If ``item`` is in a private module, but ``item`` itself is public, the parents of ``item`` are searched to see if
    ``item`` is reimported. If so, the most nested, public reimport is used.
    #. If no re-import of ``item`` is found, then a sphinx warning is emitted.
        This is likely to happen for modules missing ``__all__`` and that have external imports,
        or could be a very unlikely edge condition where a public item in a private module is reimported only by
        sibling module(s), not any (grand)parents.
    """

    # Check to see if `item` is itself private
    if item.__name__.startswith("_"):
        public_name = item.__name__
        while public_name.startswith("_"):
            public_name = public_name[1:]

        if item.__qualname__.startswith("composer"):
            log.warning(
                textwrap.dedent(f"""\
                {item.__qualname__} is private, so it should not be re-exported.
                To fix, please make it public by renaming to {public_name}"""))

    # Find and import the most nested public module of the path
    module_parts = module_name.split(".")
    public_module_parts = filter(lambda x: not x.startswith("_"), module_parts)
    public_module_name = ".".join(public_module_parts)
    public_module = importlib.import_module(public_module_name)

    # See if item is imported in `public_module`
    for name, val in vars(public_module).items():
        if val is item:
            # Found the item re-imported in `public_module` as `name`
            return f"{public_module_name}.{name}"

    # `item` was not found in `public_module`. Recursively search the parent module
    parent_module_name = ".".join(public_module_name.split(".")[:-1])
    if parent_module_name == "":
        log.warning(f"{item.__name__} in {module_name} is not re-imported by any public parent or grandparent module.")
        return None
    return determine_sphinx_path(item, parent_module_name)


def add_module_summary_tables(
    app: sphinx.application.Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: sphinx.ext.autodoc.Options,
    lines: List[str],
):
    """This hook adds in summary tables for each module, documenting all functions, exceptions, classes, and attributes.

    It links reimported imports to their original source, as not to create a duplicate, indexed toctree entry.
    It automatically inserts itself at the end of each module docstring.
    """
    del app, options  # unused
    functions: List[Tuple[str, types.FunctionType]] = []
    exceptions: List[Tuple[str, Type[BaseException]]] = []
    classes: List[Tuple[str, Type[object]]] = []
    methods: List[Tuple[str, types.MethodType]] = []
    attributes: List[Tuple[str, object]] = []
    if len(lines) == 0:
        # insert a stub docstring so it doesn't start with functions/exceptions/classes/attributes
        lines.append(name)

    if what == "module":

        try:
            all_members = list(obj.__all__)
        except AttributeError:
            all_members = list(vars(obj).keys())

        for item_name, val in vars(obj).items():
            if item_name.startswith("_"):
                # Skip private members
                continue

            if item_name not in all_members:
                # Skip members not in `__all__``
                continue

            if isinstance(val, types.ModuleType):
                # Skip modules; those are documented by autosummary
                continue

            if isinstance(val, types.FunctionType):
                functions.append((item_name, val))
            elif isinstance(val, types.MethodType):
                methods.append((item_name, val))
            elif isinstance(val, type) and issubclass(val, BaseException):
                exceptions.append((item_name, val))
            elif isinstance(val, type):
                assert issubclass(val, object)
                classes.append((item_name, val))
            else:
                attributes.append((item_name, val))
                continue

        # Sort by the reimported name
        functions.sort(key=lambda x: x[0])
        exceptions.sort(key=lambda x: x[0])
        classes.sort(key=lambda x: x[0])
        attributes.sort(key=lambda x: x[0])

        # separate hparams classes with other classes
        hparams = [(n, c) for (n, c) in classes if issubclass(c, hp.Hparams)]
        classes = [(n, c) for (n, c) in classes if not issubclass(c, hp.Hparams)]

        for category, category_name in ((functions, "Functions"), (classes, "Classes"), (hparams, "Hparams"),
                                        (exceptions, "Exceptions")):
            sphinx_lines = []
            for item_name, item in category:
                sphinx_path = determine_sphinx_path(item, item.__module__)
                if sphinx_path is not None:
                    sphinx_lines.append(f"      ~{sphinx_path}")
            if len(sphinx_lines) > 0:
                lines.append("")
                lines.append(f".. rubric:: {category_name}")
                lines.append("")
                if category_name == 'Hparams':
                    lines.append("These classes are used with :mod:`yahp` for ``YAML``-based configuration.")
                    lines.append("")
                lines.append(".. autosummary::")
                lines.append("      :nosignatures:")
                lines.append("")
                lines.extend(sphinx_lines)

        if len(attributes) > 0:
            # Documenting attributes as a list instead of a summary table because
            # an attribute's summary docstring is the type's docstring, (e.g. the docstring for dict)
            # not the docstring for the attribute.
            lines.append("")
            lines.append(f".. rubric:: Attributes")
            lines.append("")
            for item_name, item in attributes:
                lines.append(f"* :attr:`~{name}.{item_name}`")

        if len(methods) > 0:
            # Documenting bound methods as a list instead of a summary table because
            # autosummary doesn't work for bound methods
            sphinx_lines = []
            for item_name, item in methods:
                # Get the path to the class where this method is defined
                item_cls = item.__self__
                if not isinstance(item_cls, type):
                    continue
                assert issubclass(item_cls, object)
                sphinx_cls_path = determine_sphinx_path(item_cls, item_cls.__module__)
                # Get the name of this method in the class
                cls_method_name = item.__func__.__name__
                if sphinx_cls_path is not None:
                    sphinx_lines.append(f"* :meth:`~{sphinx_cls_path}.{cls_method_name}`")
            if len(sphinx_lines) > 0:
                lines.append("")
                lines.append(f".. rubric:: Methods")
                lines.append("")
                lines.extend(sphinx_lines)


def rstjinja(app, docname, source):
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # Make sure we're outputting HTML
    if app.builder.format != 'html':
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def get_algorithms_metadata() -> Dict[str, Dict[str, str]]:
    EXCLUDE = ['no_op_model']

    root = os.path.join(os.path.dirname(__file__), '..', '..', 'composer', 'algorithms')
    algorithms = next(os.walk(root))[1]
    algorithms = [algo for algo in algorithms if algo not in EXCLUDE]

    metadata = {}
    for name in algorithms:
        json_path = os.path.join(root, name, 'metadata.json')

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            for key, value in data.items():
                if key in metadata:
                    raise ValueError(f'Duplicate keys in metadata: {key}')
                metadata[key] = value

    if not metadata:
        raise ValueError(f"No metadata found, {root} not correctly configured.")
    return metadata


html_context = {'metadata': get_algorithms_metadata()}

# ClassDocumenter.add_directive_header uses ClassDocumenter.add_line to
#   write the class documentation.
# We'll monkeypatch the add_line method and intercept lines that begin
#   with "Bases:".
# In order to minimize the risk of accidentally intercepting a wrong line,
#   we'll apply this patch inside of the add_directive_header method.
# From https://stackoverflow.com/questions/46279030/how-can-i-prevent-sphinx-from-listing-object-as-a-base-class
add_line = ClassDocumenter.add_line
line_to_delete = _('Bases: %s') % u':py:class:`object`'

def add_line_no_object_base(self, text, *args, **kwargs):
    if text.strip() == line_to_delete:
        return

    add_line(self, text, *args, **kwargs)

add_directive_header = ClassDocumenter.add_directive_header

def add_directive_header_no_object_base(self, *args, **kwargs):
    self.add_line = add_line_no_object_base.__get__(self)

    result = add_directive_header(self, *args, **kwargs)

    del self.add_line

    return result

ClassDocumenter.add_directive_header = add_directive_header_no_object_base

def setup(app: sphinx.application.Sphinx):
    app.connect('autodoc-skip-member', skip_redundant_namedtuple_attributes)
    app.connect('autodoc-process-docstring', add_module_summary_tables)
    app.connect('source-read', rstjinja)
