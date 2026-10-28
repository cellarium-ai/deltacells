# Build the documentation: `make html` (the output is in build/html). Warnings are errors, so that broken links and
# references are caught in CI.

SPHINXBUILD ?= sphinx-build
SPHINXOPTS  ?= -W --keep-going
SOURCEDIR    = source
BUILDDIR     = build

.PHONY: help html clean

help:
	@echo "make html    build the documentation into build/html"
	@echo "make clean   remove build/"

html:
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
