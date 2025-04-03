# Custom Instructions for GitHub Copilot

## Project Description

N/A

## Python Code Quality Guidelines

### General Coding Principles

-   Take a test-driven-development approach
-   Follow PEP 8 style guidelines for Python code
-   Use meaningful variable and function names
-   Keep functions small and focused on a single responsibility
-   Use appropriate design patterns
-   Write modular and reusable code
-   Use type hints for function parameters and return values
-   Prefer explicit over implicit code
-   Use context managers (`with` statements) for resource management
-   Handle exceptions appropriately with specific exception types
-   Make sure to NEVER name anything that mirrors a python standard library function or class, object, etc.
    -   Example: no parameter should be named `type` or `types` use `object_type` or `object_types` instead, etc.
-   Never use `any` as a type hint, be much more specific with your type hints
-   If a function doesn't return anything, the return type should be labeled explicitly as `None`

### Google-esque Style Docstrings

-   Include docstrings for all modules, classes, methods, and functions
-   Follow Google-esque docstring format with sections for Args, Returns, Raises, Examples
-   Desired format example:

#### Module Docstrings

```python
"""
Short description of module.

Longer description of module if needed.
"""
```

#### Function Docstrings

```python
def create_parametric_line(
        self,
        sketch: adsk.fusion.Sketch | str,
        start_point: Tuple[float, float, float],
        end_point: Tuple[float, float, float],
        len_param: adsk.fusion.UserParameter | str,
        dimension_offset: float = -1.0,
        is_construction: bool = False,
    ) -> LineResult:

        """
        Create a parametric line with an associated parameter, sketch, and dimension.

        Args:
            sketch (`adsk.fusion.Sketch` | `str`): The sketch to create the line in.
                - If a string is provided, it must be the name of an existing sketch.
            start_point (`Tuple[float, float, float]`): The 3D coordinates of the start point of the line.
            end_point (`Tuple[float, float, float]`): The 3D coordinates of the end point of the line.
            len_param (`adsk.fusion.UserParameter` | `str`): The parameter controlling the length of the line.
                - If a string is provided, it must be the name of an existing parameter.
            dimension_offset (`float`): The offset of the dimension text from the line.
                - Defaults to -1.0.
            is_construction (`bool`): Whether the line is a construction line.
                - Defaults to False.

        Returns:
            `ParametricLineResult`: A dataclass containing the created sketch, line, and parameter.
        """
```

#### Class Docstrings

```python
class ClassName:
    """
    Short description of class.

    Longer description of class if needed.

    Attributes:
        attribute_1 (`type_of_attribute_1`): Description of attribute_1
        attribute_2 (`type_of_attribute_2`): Description of attribute_2

    Methods:
        `method_1`: Description of method_1
        `method_2`: Description of method_2
    """

    def __init__(self, param_1: type, param_2: type) -> None:
        """
        Short description of method.

        Longer description of method if needed.

        Args:
            `param_1` (type_of_param_1): Description of param_1
            `param_2` (type_of_param_1): Description of param_2
        """
```

### Test-Driven Development

-   Write tests before implementing functionality
-   Use pytest for testing
-   Tests should be in a separate `tests` directory with the same structure as the source code
-   Test file names should follow the pattern `test_*.py`
-   Each test should focus on one specific aspect of functionality
-   Include both positive and negative test cases
-   Use fixtures and parameterization for efficient testing
-   Aim for high test coverage (>80%)

### Code Organization

-   Follow a clear module structure
-   Use appropriate design patterns
-   Keep classes cohesive and decoupled
-   Prefer composition over inheritance
-   Use packages to group related modules
-   Use `__init__.py` files to define package structure
-   Organize imports in the following order:
    1. Standard library imports
    2. Related third-party imports
    3. Local application/library specific imports
-   Separate imports with a blank line between each group

### Error Handling

-   Use specific exception types
-   Provide informative error messages
-   Log errors appropriately
-   Fail gracefully when possible
-   Use the rich library for better error messages and its logging capabilities

### Performance Considerations

-   Consider time and space complexity when implementing algorithms
-   Avoid unnecessary computations
-   Use appropriate data structures
-   Consider memory usage for large datasets

### Security Best Practices

-   Never hard-code sensitive information
-   Sanitize inputs to prevent injection attacks
-   Use safe methods for serialization/deserialization
-   Follow least privilege principle

### Readability

-   Include explanations of complex logic in docstrings, only use comments when absolutely necessary
-   Use consistent naming conventions
-   Break long lines for better readability
-   Use whitespace effectively to group related code
