# Feature-Level Working Spec Guide

## Overview

For better audit trails and feature isolation, each feature should have its own working spec file in `.caws/` with the naming pattern `FEAT-XXX.yaml`.

## When to Create a Feature Spec

Create a new feature spec when:
- Starting work on a discrete, scoped feature
- The feature has its own acceptance criteria
- The feature will be completed in one or more focused PRs
- You want isolated tracking and audit trail

## File Naming Convention

- **Feature specs**: `FEAT-001.yaml`, `FEAT-002.yaml`, etc.
- **Bug fixes**: `FIX-001.yaml`, `FIX-002.yaml`, etc.
- **Architecture changes**: `ARCH-001.yaml`, etc.
- **Project baseline**: `working-spec.yaml` (PROJ-185)

## Feature Spec Structure

Each feature spec should include:

1. **Unique ID**: `FEAT-XXX` format
2. **Scoped boundaries**: Only files/modules touched by this feature
3. **Feature-specific acceptance criteria**: Clear, testable outcomes
4. **Appropriate change budget**: Based on feature scope
5. **Risk tier**: Appropriate for the feature's impact
6. **Rollback plan**: How to revert if needed

## Workflow

### 1. Create Feature Spec

```bash
# Copy example template
cp .caws/FEAT-001.example.yaml .caws/FEAT-002.yaml

# Edit with feature-specific details
vim .caws/FEAT-002.yaml

# Validate
caws validate .caws/FEAT-002.yaml
```

### 2. Implement Feature

- Reference feature spec for acceptance criteria
- Stay within `scope.in` boundaries
- Respect change budget limits
- Write tests for each acceptance criterion

### 3. Track Progress

```bash
# Update acceptance criterion status
caws progress update --criterion-id="A1" --status="completed" --tests-passing=5

# Check overall feature progress
caws status --spec-file=.caws/FEAT-002.yaml
```

### 4. Validate Before PR

```bash
# Validate feature spec
caws validate .caws/FEAT-002.yaml

# Run quality gates
caws quality-gates

# Verify acceptance criteria met
caws evaluate --spec-file=.caws/FEAT-002.yaml
```

## Example: Process Supervision Feature

See `.caws/FEAT-001.example.yaml` for a complete example based on a real feature from the distillation roadmap.

## Relationship to Project Spec

- **Project spec** (`working-spec.yaml`): High-level project goals, overall scope, project-wide invariants
- **Feature specs** (`FEAT-XXX.yaml`): Specific feature implementation details, scoped acceptance criteria, feature-specific boundaries

Both can coexist - feature specs inherit project-level constraints but add feature-specific details.

## Benefits

- **Clear audit trail**: Each feature has its own spec and history
- **Isolated tracking**: Progress tracked per feature
- **Better provenance**: Changes linked to specific feature specs
- **Easier reviews**: Reviewers see exactly what the feature should accomplish
- **Rollback clarity**: Each feature has its own rollback plan

