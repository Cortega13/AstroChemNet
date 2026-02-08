# AGENTS.md

# Coding Principles

always use YAGNI + SOLID + DRY + KISS
before and after your code.

# Documentation

- Every class and function MUST have a one-line docstring.
- Every script must have a 1 liner at the top describing the purpose.
- Backwards compatability does not matter at this moment because we are in development stage.

# Coding Standards

-   Use only single-line docstrings that describe what the function does, not how it does it
-   Keep docstrings concise and helpful
-   Remove all Args, Returns, Raises sections from docstrings
-   Functions must be at most 30 lines long
-   Functions should not be 1 liners.
-   I hate having too many input args. Only use input args that are necessary for the script or I asked for.
-   Functions should flow downwards based on usage in the main function - helper functions should be defined before they're called
-   Main functions should be highly readable through descriptive function names that clearly indicate the flow of operations
-   All function inputs and outputs must have type annotations
-   Types must be descriptive - use dataclasses, TypedDict, or type aliases for complex types
-   Use comments sparingly. I like them inline, but I like them to be very concise and primarly for explaining multiple lines of code not just 1.
-   Always include an example usage line in the script docstring. Additionally, include a list of assumptions when you wrote the code or made modifications to it.
-   All code should be super easy to read and understand. It should be direct, concise, and efficient.


Typing tips:
- OmegaConf is not a type. The correct type for it is DictConfig, ListConfig which must be importated like from omegaconf import DictConfig, ListConfig, OmegaConf.
