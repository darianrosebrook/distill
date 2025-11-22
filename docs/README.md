# Architecture & Documentation Index

This directory contains architectural documentation, system design, and core integration guides for the kimi-student project.

## System Architecture

### Core Concepts

- **[ARBITER_THEORY.md](ARBITER_THEORY.md)** - Arbiter stack architecture, requirements, and theoretical foundations
- **[LATENT_REASONING.md](LATENT_REASONING.md)** - Latent reasoning capabilities and implementation patterns

### Model Documentation

- **[8_BALL_MODEL_CARD.md](8_BALL_MODEL_CARD.md)** - 8-Ball model specifications, capabilities, and characteristics
- **[DATASET_CARD_CONTEXTUAL.md](DATASET_CARD_CONTEXTUAL.md)** - Dataset schema, usage guidelines, and generation policies

## Integration & Workflows

### Knowledge Distillation

- **[DISTILLATION_GUIDE.md](DISTILLATION_GUIDE.md)** - Knowledge distillation workflow and integration
- **[PRODUCTION_DISTILLATION_WORKFLOW.md](PRODUCTION_DISTILLATION_WORKFLOW.md)** - Production distillation pipeline

### Dataset Management

- **[CONTEXTUAL_DATASET_GENERATION.md](CONTEXTUAL_DATASET_GENERATION.md)** - Contextual dataset generation workflow

## Development & Testing

### Testing Strategy

- **[TOY_TEST_STRATEGY.md](TOY_TEST_STRATEGY.md)** - Lightweight testing approach and test doubles
- **[MUTATION_TESTING.md](MUTATION_TESTING.md)** - Mutation testing framework and strategies
- **[SCALE_TESTS.md](SCALE_TESTS.md)** - Scale testing methodology (N=1k, N=10k)

### Development Workflow

- **[LOCAL_DEVELOPMENT_WORKFLOW.md](LOCAL_DEVELOPMENT_WORKFLOW.md)** - Local development setup and workflow
- **[CAWS_AGENT_GUIDE.md](CAWS_AGENT_GUIDE.md)** - CAWS agent integration for AI-assisted development

## Operations

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment architecture and operational procedures

## Documentation Organization

This repository organizes documentation into four categories:

1. **Key Architectural Docs** (root or `docs/`) - Standard repository documentation covering architecture, workflows, and guides
2. **Internal Documents** (`docs/internal/`) - Implementation plans, detailed analysis, and development notes
3. **External Documentation** (`docs/external/`) - Sourced documentation from outside the repository
4. **Archive** (`docs/archive/`) - Temporal documents, status reports, completion summaries, and potentially outdated content

## External Resources

See [external/](external/) for public-facing documentation including Kimi K2 setup and integration guides.

## Internal Development

See [internal/](internal/) for implementation plans, detailed technical analysis, and development notes.

## Historical Documentation

See [archive/](archive/) for temporal status reports, completion summaries, and historical documents.

## Related Documentation

- **Test Suite**: [../tests/README.md](../tests/README.md) - Test organization and execution
- **Project Status**: [../docs-status/README.md](../docs-status/README.md) - Project management documentation (git-ignored)
- **Main README**: [../README.md](../README.md)
- **Changelog**: [../CHANGELOG.md](../CHANGELOG.md)
- **Commit Conventions**: [../COMMIT_CONVENTIONS.md](../COMMIT_CONVENTIONS.md)
