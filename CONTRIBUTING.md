# Contributing to ACTUNEO

Thank you for your interest in contributing to ACTUNEO! 🎉

ACTUNEO is an open-source, community-driven actuarial Python library that empowers African and Zimbabwean actuaries. We welcome contributions from actuaries, developers, students, and anyone passionate about making actuarial tools more accessible.

## Ways to Contribute

### 🐛 Report Issues
- Found a bug? Have a feature request? [Open an issue](https://github.com/yourusername/actuneo/issues)
- Use clear, descriptive titles
- Provide steps to reproduce bugs
- Include your environment details (Python version, OS, etc.)

### 💡 Suggest Features
- Have ideas for new modules or improvements?
- Check existing issues first to avoid duplicates
- Describe the problem you're solving and your proposed solution

### 🔧 Code Contributions
- Fix bugs, add features, improve documentation
- Follow our development workflow below

### 📚 Documentation
- Improve existing documentation
- Translate documentation
- Create tutorials and examples

### 🧪 Testing
- Write and improve tests
- Help maintain test coverage

### 🌍 Community Building
- Help answer questions in issues and discussions
- Share ACTUNEO with other African actuaries
- Organize meetups or workshops

## Development Workflow

### 1. Fork and Clone
```bash
git clone https://github.com/yourusername/actuneo.git
cd actuneo
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 4. Make Your Changes
- Follow our [Coding Standards](#coding-standards)
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 5. Commit Your Changes
```bash
git add .
git commit -m "feat: add new mortality table validation

- Add input validation for mortality tables
- Improve error messages
- Add comprehensive tests"
```

### 6. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```
Then create a Pull Request on GitHub.

## Coding Standards

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Commit Messages
We follow [Conventional Commits](https://conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing changes
- `chore`: Maintenance tasks

Examples:
```
feat(mortality): add Gompertz-Makeham mortality model
fix(finance): correct duration calculation for zero rates
docs: update installation instructions
```

### Testing
- Write tests for all new functionality
- Maintain test coverage above 80%
- Run tests before committing:
```bash
pytest
```

### Documentation
- Use docstrings for all public functions and classes
- Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings
- Update documentation for API changes
- Add examples for complex functionality

## African Actuarial Context

ACTUNEO is built with African actuarial realities in mind. When contributing:

### 🎯 Focus Areas
- **Localized Data**: Contribute mortality tables, inflation data, and economic indicators for African markets
- **Regulatory Compliance**: Add support for regional regulatory frameworks (IPEC Zimbabwe, PASA, SAM, etc.)
- **Cultural Relevance**: Consider informal sector, microinsurance, and low-coverage environments
- **Language**: Support multiple languages where appropriate

### 📊 Data Contributions
- Share anonymized actuarial data (with permission)
- Contribute country-specific economic indicators
- Validate models against African market data

### 🤝 Cultural Sensitivity
- Be respectful of diverse cultural contexts
- Consider local business practices and traditions
- Ensure inclusive language and examples

## Code Review Process

### Pull Request Guidelines
- Provide clear description of changes
- Reference related issues
- Include screenshots for UI changes
- Ensure CI/CD checks pass

### Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes without discussion
- [ ] Performance impact considered
- [ ] Security implications reviewed

### Review Process
1. Automated checks (linting, tests, coverage)
2. Peer review by maintainers
3. Domain expert review for actuarial accuracy
4. Final approval and merge

## Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Email**: [Your contact email]

### Community Guidelines
- Be respectful and inclusive
- Help newcomers learn and contribute
- Focus on constructive feedback
- English is the primary language, but we welcome translations

## Recognition

Contributors are recognized through:
- GitHub contributor statistics
- Mention in release notes
- ACTUNEO community acknowledgments
- Co-authorship on academic papers (where applicable)

## License

By contributing to ACTUNEO, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ACTUNEO and helping build the future of actuarial science in Africa! 🇿🇼✨
