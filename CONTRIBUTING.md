# Contributing to Faith as the First Law

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing experimental protocols, code, and analyses.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### 1. Reporting Issues
- **Bug reports**: Include steps to reproduce, expected vs. actual behavior, and environment details
- **Feature requests**: Describe the use case and why it's valuable for the research
- **Research questions**: For scientific discussions, use GitHub Discussions

### 2. Contributing Code
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-experiment`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Run existing tests**: `pytest tests/`
7. **Commit your changes**: Use descriptive commit messages
8. **Push to your fork**: `git push origin feature/amazing-experiment`
9. **Open a Pull Request**

### 3. Contributing Experiments
New experimental protocols should include:
- A clear hypothesis statement
- Detailed methodology
- Data collection templates
- Analysis scripts
- Example data
- Expected results
- Falsification criteria

### 4. Contributing Documentation
- Fix typos or clarify explanations
- Add examples
- Improve formatting
- Translate to other languages

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/scottdevine/faith-as-first-law.git
   cd faith-as-first-law
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .[dev,full]
   ```

4. Run tests:
   ```bash
   pytest tests/
   ```

## Coding Standards

### Python Code
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where appropriate
- Document functions with docstrings
- Maximum line length: 88 characters (Black formatting)

### R Code
- Follow [tidyverse style guide](https://style.tidyverse.org/)
- Use meaningful variable names
- Comment complex sections

### Markdown/LaTeX
- Use proper formatting
- Include equations in LaTeX when needed
- Link to references appropriately

## Experiment Quality Standards

All experiments must be:
1. **Falsifiable**: Clear criteria for rejection
2. **Reproducible**: Detailed protocols anyone can follow
3. **Ethical**: Respect participant rights and privacy
4. **Transparent**: All data and code openly available
5. **Statistically sound**: Appropriate sample sizes and tests

## Pre-commit Checklist

Before submitting a PR:
- [ ] Code passes all tests
- [ ] Documentation is updated
- [ ] No sensitive data included
- [ ] Commit messages are descriptive
- [ ] Changes are focused and minimal

## Review Process

1. **Automated checks**: GitHub Actions will run tests
2. **Maintainer review**: At least one maintainer will review
3. **Scientific review**: For experimental protocols, scientific accuracy will be checked
4. **Merge**: Once approved, changes will be merged

## Recognition

All contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in relevant papers
- Given commit credit in the repository

## Questions?

- Open an issue for technical questions
- Use GitHub Discussions for scientific discussions
- Email scottdevine01@gmail.com for private correspondence

Thank you for contributing to open, reproducible science!
