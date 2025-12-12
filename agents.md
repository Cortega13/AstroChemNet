# Coding Principles

always use YAGNI + SOLID + DRY + KISS
before and after your code.

# Documentation

- Every class and function MUST have a one-line docstring.
- Every script must have a 1 liner at the top describing the purpose.


# Coding Standards

-   Use only single-line docstrings that describe what the function does, not how it does it
-   Keep docstrings concise and helpful
-   Remove all Args, Returns, Raises sections from docstrings
-   Functions must be at most 30 lines long
-   Functions should flow downwards based on usage in the main function - helper functions should be defined before they're called
-   Main functions should be highly readable through descriptive function names that clearly indicate the flow of operations
-   All function inputs and outputs must have type annotations
-   Types must be descriptive - use dataclasses, TypedDict, or type aliases for complex types



- Do not use comments anywhere. I only like them in docstrings.
