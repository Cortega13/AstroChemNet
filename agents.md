# minor-fixes

This is a rule for refactor's/minor fixes throughout the codebase.

## Guidelines

- I will copy snippets of code into your chat. I will provide a comment about it that I dislike.
- You will then read this code, and read what leads to it.
- You will then generate a .md file in the plans folder which shows a brainstorm of solutions to the problem.
- The solutions must aim to follow minimization rules, as in try to perhaps solve upstream issues to make it more coherent throughout the codebase.



1. Clarity Over Cleverness

Minimal code is not code-golf.

Bad minimalism:

print(max(map(len,s.split())))

Better minimalism:

words = s.split()
longest = max(len(w) for w in words)
print(longest)

👉 The rule: Shortest that remains obvious.

2. KISS (Keep It Simple, Stupid)

Avoid abstraction until necessary

Avoid generalization until required

Solve today’s problem, not imaginary future ones

Minimalism often means:

Fewer layers

Fewer classes

Fewer patterns

Fewer dependencies

3. YAGNI (You Aren’t Gonna Need It)

Don’t build:

Config systems “just in case”

Plug-in architectures prematurely

Over-general utility layers

If the feature isn’t required now, don’t code it.

4. DRY — But Not at the Cost of Clarity

Don’t repeat logic unnecessarily.

But:

Avoid over-abstracting tiny duplication.

Sometimes repeating 3 lines is clearer than inventing a utility.

Minimal code balances DRY with readability.

5. High Signal-to-Noise Ratio

Every line should justify its existence.

Ask:

Does this variable add clarity?

Does this abstraction reduce complexity?

Can this logic be expressed more directly?

Delete:

Unused variables

Over-commented obvious code

Redundant wrappers

6. Prefer Composition Over Inheritance

Inheritance often introduces complexity.

Minimalism favors:

Small pure functions

Simple data structures

Composition of behaviors

7. Use the Language Idiomatically

Minimal code uses built-in tools well.

Example:

Verbose:

result = []
for item in items:
    if item.active:
        result.append(item.name)

Minimal & idiomatic:

result = [item.name for item in items if item.active]

Leverage:

Standard library

Built-in functions

Native language patterns

8. Minimize State

State is where complexity grows.

Prefer:

Pure functions

Stateless transformations

Immutable data when practical

Less state = less code needed to manage it.

9. Reduce Surface Area

Minimalist design often means:

Fewer public functions

Smaller APIs

Smaller modules

Fewer parameters

Ask:

Can this interface be smaller?

10. Refactor by Deletion

Minimalist developers constantly ask:

What can be removed?

What can be merged?

What can be simplified?

Often the best refactor removes code rather than adding structure.
