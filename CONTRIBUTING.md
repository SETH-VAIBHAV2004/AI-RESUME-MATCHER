# 🤝 Contributing to Resume Matcher

Thank you for your interest in contributing to Resume Matcher! We welcome contributions from the community and are excited to see what you'll bring to the project.

## 🌟 Ways to Contribute

### 🐛 Bug Reports
- Found a bug? [Create an issue](https://github.com/yourusername/resume-matcher/issues/new?template=bug_report.md)
- Include detailed steps to reproduce
- Provide system information and error messages
- Add screenshots if applicable

### ✨ Feature Requests
- Have an idea? [Submit a feature request](https://github.com/yourusername/resume-matcher/issues/new?template=feature_request.md)
- Describe the problem you're trying to solve
- Explain your proposed solution
- Consider the impact on existing users

### 📝 Documentation
- Improve README, code comments, or docstrings
- Add examples and tutorials
- Fix typos and clarify explanations
- Translate documentation to other languages

### 🧪 Testing
- Add unit tests for new features
- Improve test coverage
- Test on different platforms and Python versions
- Validate with real-world datasets

## 🚀 Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- 8GB RAM (for BERT models)

### Setup Steps
```bash
# 1. Fork and clone the repository
git clone https://github.com/SETH-VAIBHAV2004/AI-RESUME-MATCHER
cd resume-matcher

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# 4. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Download required models
python -m spacy download en_core_web_sm

# 6. Run tests to verify setup
python test_system.py
```

## 📋 Development Guidelines

### Code Style
- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use meaningful variable and function names
- Add type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible

### Example Code Style
```python
def extract_skills_hybrid(self, text: str, exact_threshold: int = 90, 
                         fuzzy_threshold: int = 75) -> Dict[str, List[Tuple[str, str, int]]]:
    """
    Extract skills using hybrid approach (exact + fuzzy matching).
    
    Args:
        text (str): Input text to analyze
        exact_threshold (int): Threshold for considering exact matches
        fuzzy_threshold (int): Threshold for fuzzy matches
        
    Returns:
        Dict[str, List[Tuple[str, str, int]]]: Skills with (skill, match_type, score)
    """
    # Implementation here
    pass
```

### Testing
- Write unit tests for new functions
- Ensure tests pass on multiple Python versions
- Test edge cases and error conditions
- Aim for >80% code coverage

### Commit Messages
Use conventional commit format:
```
type(scope): description

feat(skill-extraction): add fuzzy matching for skill variants
fix(bert): resolve memory leak in batch processing
docs(readme): update installation instructions
test(api): add integration tests for CLI interface
```

## 🔄 Pull Request Process

### Before Submitting
1. **Create an issue** first to discuss major changes
2. **Fork the repository** and create a feature branch
3. **Write tests** for your changes
4. **Update documentation** as needed
5. **Run the test suite** and ensure all tests pass
6. **Follow code style guidelines**

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
python test_system.py

# Run specific component tests
python -m pytest tests/test_skill_extraction.py -v

# Run with coverage
python -m pytest --cov=resume_matcher tests/

# Performance benchmarks
python benchmark_models.py
```

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Speed and memory benchmarks
- **End-to-End Tests**: Complete workflow validation

## 📊 Performance Considerations

### Model Performance
- Maintain >85% accuracy on test datasets
- Keep processing time under 1 second for standard documents
- Monitor memory usage, especially for BERT models
- Test with various document sizes and formats

### Code Performance
- Profile code for bottlenecks
- Use appropriate data structures
- Cache expensive computations
- Consider parallel processing for batch operations

## 🎯 Priority Areas for Contribution

### High Priority
- **Multi-language Support**: Extend to non-English resumes
- **Model Optimization**: Improve speed and accuracy
- **Industry Specialization**: Domain-specific skill dictionaries
- **API Development**: RESTful API for integrations

### Medium Priority
- **Advanced Visualizations**: Interactive charts and insights
- **Batch Processing**: Improved multi-document handling
- **Export Features**: Additional report formats
- **Mobile Optimization**: Responsive design improvements

### Low Priority
- **UI Enhancements**: Visual improvements
- **Documentation**: Additional examples and tutorials
- **Testing**: Expanded test coverage
- **Refactoring**: Code organization improvements

## 🏆 Recognition

### Contributors
All contributors will be recognized in:
- README.md contributors section
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Annual contributor appreciation posts

### Contribution Types
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation
- 🧪 Testing
- 🎨 Design
- 💡 Ideas and feedback
- 🌍 Translation
- 📊 Data and research

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: [maintainers@resumematcher.com](mailto:maintainers@resumematcher.com)
- **Discord**: [Join our community](https://discord.gg/resumematcher)

### Response Times
- **Bug reports**: Within 48 hours
- **Feature requests**: Within 1 week
- **Pull requests**: Within 1 week
- **Questions**: Within 24 hours

## 📜 Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Expected Behavior
- Be respectful and constructive in all interactions
- Welcome newcomers and help them get started
- Focus on what's best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Spam or off-topic discussions
- Sharing private information without permission

### Enforcement
Violations of the code of conduct should be reported to [conduct@resumematcher.com](mailto:conduct@resumematcher.com). All reports will be reviewed and investigated promptly.

## 🎉 Thank You!

Your contributions make Resume Matcher better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, every contribution matters.

**Happy coding!** 🚀

---

*For questions about contributing, please [create an issue](https://github.com/yourusername/resume-matcher/issues) or reach out to the maintainers.*