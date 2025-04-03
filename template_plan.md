# Template System Implementation Plan

## Key Implementation Decisions

### 1. File-Based Templates vs. Hard-Coded Strings

**Decision**: Store templates as actual files rather than hard-coded strings in code.

**Reasoning**:

-   Separation of concerns - content should be separate from code
-   Easier editing without code changes
-   Templates can be updated without redeploying application
-   Better readability in dedicated template files
-   Templates can be managed by non-developers

**Implementation**:

-   Created a `memcp/templates/` directory with subdirectories for different template types
-   Used `.txt` files for storing template content without any special formatting requirements
-   Templates are plain text with Python string formatting placeholders

### 2. Robust Template Resolution

**Decision**: Implement a multi-layer resolution strategy with fallbacks.

**Reasoning**:

-   Templates should be found regardless of how the application is installed or run
-   Users should be able to override default templates with custom ones
-   System should never fail catastrophically if a template is missing

**Implementation**:

-   Search order:
    1. Explicit path (if absolute)
    2. Current working directory
    3. User-specified template directory from config
    4. Package directory
    5. Importlib resources (for installed packages)
    6. Fallback to built-in string templates

### 3. Cross-Platform and Installation-Method Compatibility

**Decision**: Support both file system paths and package resources.

**Reasoning**:

-   Application may be run directly from source code
-   Application may be installed as a package
-   Must work across different operating systems
-   Should work in containerized environments

**Implementation**:

-   Used `importlib.resources` for accessing package resources
-   Support for both `Path` and `Traversable` objects
-   Different loading mechanisms depending on the resource type
-   Proper handling of file system access errors

### 4. Security Considerations

**Decision**: Implement template sanitization and size limits.

**Reasoning**:

-   Templates with user input can be a source of security vulnerabilities
-   Format string attacks could expose internal data
-   Very large templates could cause memory issues
-   Input validation is always important

**Implementation**:

-   Added 100KB size limit for templates
-   Sanitize templates to prevent attribute access via format strings
-   Validate required variables before using templates
-   Proper error handling for malformed templates

### 5. Configuration Flexibility

**Decision**: Allow template configuration via TOML and provide sensible defaults.

**Reasoning**:

-   Users should be able to customize template behavior
-   Defaults should work out of the box
-   Configuration should be centralized and consistent

**Implementation**:

-   Added `templates` section to config.toml
-   Added template-related fields to MemCPServerConfig
-   Created a dedicated TemplateConfig class
-   Support for environment variable overrides

### 6. Error Handling and Fallbacks

**Decision**: Implement a multi-stage fallback mechanism.

**Reasoning**:

-   Template failures should not crash the application
-   Fallback to simpler templates is better than no template
-   The system should be resilient to configuration errors

**Implementation**:

-   Try specific template → try default template → try built-in template → use minimal string
-   Clear error messages when templates fail to load
-   Proper logging of template-related errors
-   Graceful degradation path

### 7. Type Safety and Modern Python Features

**Decision**: Use strong typing and modern Python features.

**Reasoning**:

-   Type hints improve IDE support and code readability
-   Proper types catch errors at development time
-   Support for modern Python patterns

**Implementation**:

-   Used type hints throughout the code
-   Handle both Pydantic V1 and V2 compatibility
-   Used Optional types and Union types correctly
-   Generic TypeVar for model loading

## Template System Architecture

```
memcp/
├── config/
│   ├── templates.py    # Template loading and formatting logic
│   ├── settings.py     # TemplateConfig definition
│   └── loader.py       # TOML configuration loading
├── server/
│   ├── mcp.py          # MCP server with _get_instructions using templates
│   └── models.py       # MemCPServerConfig with template settings
└── templates/          # Template files
    ├── instructions/   # Server instruction templates
    │   ├── default_instructions.txt
    │   └── minimal_instructions.txt
    └── README.md       # Documentation for template system
```

## Template Formatting

Templates use Python's standard string formatting syntax with named placeholders:

```
Welcome to {server_name} - a memory service for AI agents.

{server_description}

This server provides memory capabilities through a graph-based knowledge store.
```

## Best Practices Applied

1. **No textwrap.dedent() for Files**

    - Template files should be properly formatted in the file
    - Dedent is only useful for multi-line strings in code, not for files
    - Files should be read as-is to preserve intentional formatting

2. **Proper Error Handling**

    - Specific exception types for template errors
    - Fallback mechanisms at multiple levels
    - Proper logging of errors
    - Never crashing the application due to template issues

3. **Security-First Design**

    - Template sanitization
    - Size limits
    - Path validation
    - Safe fallbacks

4. **Using importlib.resources Correctly**

    - Handle both Path and Traversable objects
    - Properly access package resources
    - Work in both development and installed modes

5. **Modular API Design**

    - Clear separation of concerns
    - Well-defined function boundaries
    - Reusable components

6. **Pydantic Integration**

    - Leverage Pydantic for configuration validation
    - Type-safe configuration
    - Compatible with both V1 and V2

7. **Documentation**
    - Clear docstrings
    - Type hints
    - README for template directory
    - Comments for complex logic

## Key Functions

1. **get_template_path**: Resolves template file paths with a multi-layer strategy
2. **load_template**: Loads and validates a template with proper error handling
3. **format_template**: Formats a template with provided variables
4. **sanitize_template**: Prevents dangerous format string patterns
5. **validate_template**: Ensures a template has all required variables

## TOML Configuration Structure

```toml
[templates]
templates_dir = ""  # Optional, uses default locations if empty
instructions_template_name = "default_instructions"
validate_templates = true
default_template_name = "default_instructions" # Fallback template

[mcp]
# Template settings can also be specified here
templates_dir = ""
instructions_template_name = "default_instructions"
```

## Generic Type in loader.py Explanation

The generic type `T` in `load_model_from_toml` serves several important purposes:

1. **Type Safety**: It ensures the return type is exactly the same as the model_cls parameter type
2. **IDE Autocompletion**: IDEs can provide proper autocompletion for the returned model
3. **Code Clarity**: Makes it clear that the function returns the same type that was passed in
4. **Error Prevention**: Prevents accidentally assigning the result to a variable of the wrong type

Without the generic type, the function would either need to return `Any` (losing type safety) or `BaseModel` (losing specific model information).

## Template Validation Process

1. Check if template exists at resolved path
2. Verify template size is within limits
3. Read template content
4. Sanitize template to remove dangerous patterns
5. Extract and validate required variables
6. Test format with dummy values
7. Return validated template string

## Implementation Improvements Over Original Plan

1. **Better Fallback System**: Added multi-stage fallbacks including built-in template strings
2. **importlib.resources Support**: Added support for package resources through importlib
3. **Type Safety**: Improved type hints and proper handling of different resource types
4. **Security Enhancements**: More thorough template sanitization and validation
5. **Error Handling**: More comprehensive error handling with specific error types
6. **Configuration Integration**: Better integration with existing configuration system
7. **Logger Support**: Integrated with proper logging system

## Final Recommendations

1. Consider adding template caching for performance in production environments
2. Add more template categories as needed (beyond instructions)
3. Consider supporting markdown or other formats for richer templates
4. Add unit tests specifically for template edge cases
5. Consider adding template versioning for sensitive templates
