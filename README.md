# Template Github Repository
[![Build](https://github.com/edmundsj/sugarplot/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/edmundsj/sugarplot/actions/workflows/python-package-conda.yml) [![docs](https://github.com/edmundsj/sugarplot/actions/workflows/build-docs.yml/badge.svg)](https://github.com/edmundsj/sugarplot/actions/workflows/build-docs.yml) [![codecov](https://codecov.io/gh/edmundsj/sugarplot/branch/main/graph/badge.svg?token=C7U4y4Gihv)](https://codecov.io/gh/edmundsj/sugarplot)

This is a template repository for python projects which use sphinx for
documentation, github actions for building, pytest and codecov for test
coverage.


## Getting Started
### Installation via pip

```
pip install sugarplot
```

## Features

- Github actions unit test integration via pytest
- Github actions package management with conda
- Github actions documentation build using sphinx and reST/markdown, with auto
self-push to repository after successful build
- Github pages documentation hosting/integration
- Local commits hooks run full test suite
- Coverage uploaded automatically to codecov after successful build
- [FUTURE] Auto-deploy to pyPi/testpyPi after successful build

## Common Issues
- Re-running builds on github actions will cause them to fail, as the build number deployed to PyPi depends on the github run number, which does not change if you restart a build.
- Pypi deploy is a little slower than test pypi, so it may not always be downloading the latest version.


## How to Use
### Adding Additional Unit Tests
- Any time you want to add additional unit tests just add them to those in the
``tests/`` directory and prepend with the name ``test``. These will be
automatically found by pytest and run during local commits and remote builds.

### Writing the Documentation
- The documentation source is located in ``docs/source`` and is written in
restructured text (markdown is also available).

### Building the Documentation
Simply run ``make html`` from the ``docs/`` directory. This will compile the
files in the ``docs/source/`` directory, and place them in the main ``docs/``
directory where github pages can find them.

## Dependencies / Technologies Used
- [Sphinx](http://www.sphinx-doc.org/)
- [pytest](https://docs.pytest.org/en/stable/index.html)
- [Github Actions](https://github.com/features/actions)
- [Codecov](https://codecov.io/)
- [Github Pages](https://pages.github.com/)

## Acknowledgements
Thanks to all the great people on stack overflow and github, for their
seemingly boundless tolerance to my and others' questions. 
