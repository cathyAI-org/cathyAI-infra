# cathyAI Development Rules

## Code Style and Documentation

### PEP 8 Compliance
- Follow PEP 8 style guidelines to the greatest extent possible
- Use 4 spaces for indentation
- Line length limit of 88 characters (Black formatter standard)
- Use snake_case for functions and variables
- Use PascalCase for classes

### Documentation Standards
- Use reStructuredText (reST) format for all new function docstrings
- Include parameter types and return types in docstrings
- Document exceptions that may be raised

Example docstring format:
```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    :param param1: Description of param1
    :type param1: str
    :param param2: Description of param2
    :type param2: int
    :return: Description of return value
    :rtype: bool
    :raises ValueError: When invalid input is provided
    """
```

## Commit Message Structure

All commits must follow this exact format:

```
[type]: Title

Description:

list of changes, one short line per change. each entry on the list should be 

  - [entry]

Additional info:

additional info about the changes that might be good to know. 

  - [info]
```

### Commit Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Post-Change Requirements

### Mandatory Updates
- **After each change**: Update README.md and tests to reflect the latest implementation
- **Documentation Sync**: Keep README synchronized with actual code behavior and file structure
- **Test Coverage**: Ensure tests validate new functionality and structural changes
- **Pre-commit Testing**: Run all tests before committing to verify no issues are introduced

### Implementation Standards
- **Minimal Code**: Write only the absolute minimal amount of code needed to address requirements correctly
- **No Verbose Implementations**: Avoid any code that doesn't directly contribute to the solution
- **Validation**: All changes must pass existing tests and maintain system functionality

## Repository-Specific Standards

### Service Architecture
- Maintain independence between webbui_chat and characters_api services
- Use shared resources (characters/, public/) consistently across services
- Follow Docker containerization patterns for both services

### Configuration Management
- Use .env.template files for all services
- Maintain consistent environment variable naming
- Document all configuration options in README

### Testing Requirements
- All shared resources must have corresponding tests in tests/test_app.py
- Service-specific functionality must have dedicated test files
- Tests must validate file structure, API contracts, and configuration consistency