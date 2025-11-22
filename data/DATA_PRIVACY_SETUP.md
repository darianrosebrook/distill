# Data Privacy Setup

## Overview

This repository contains **only the infrastructure** for data processing (schemas, generators, scripts). The actual dataset files (`.jsonl`, `.json`) generated from paid API calls are kept in a **separate private git repository** to protect your investment.

## Current Status

All data files are git-ignored in this repository. They should be tracked in your private data repository instead.

## Setting Up Your Private Data Repository

### Option 1: Separate Private Repository (Recommended)

1. **Create a private git repository** (e.g., `distill-data-private`):
   ```bash
   # On GitHub/GitLab, create a new private repository
   # Then clone it locally:
   git clone <private-repo-url> ../distill-data-private
   ```

2. **Copy your data files** to the private repository:
   ```bash
   # From the main repo root:
   cp -r data/*.jsonl data/*.json ../distill-data-private/ 2>/dev/null || true
   cp -r data/judge/*.jsonl data/judge/*.json ../distill-data-private/judge/ 2>/dev/null || true
   cp -r data/logits/*.json ../distill-data-private/logits/ 2>/dev/null || true
   # ... copy other data files as needed
   ```

3. **Commit to your private repository**:
   ```bash
   cd ../distill-data-private
   git add .
   git commit -m "Initial private data repository"
   git push origin main
   ```

### Option 2: Git Submodule (For Integration)

If you want to link your private data repository to this repo (but keep it private):

1. **Add as a submodule**:
   ```bash
   # Add your private repo as a submodule
   git submodule add <private-repo-ssh-url> data/private
   git submodule update --init --recursive
   ```

2. **Update paths in scripts** to use `data/private/` instead of `data/`

3. **Clone with submodules** (for new clones):
   ```bash
   git clone --recurse-submodules <repo-url>
   ```

### Option 3: Local-Only (No Git)

Keep data files local-only (not in git at all):

- Data files remain in `data/` locally
- They're already git-ignored
- Backup manually or use external backup solution
- No git tracking of data files

## Removing Already-Tracked Files

If you have data files currently tracked in this repository, remove them from git (but keep locally):

```bash
# Remove from git tracking (files stay on disk)
git rm --cached data/*.jsonl data/*.json
git rm --cached data/**/*.jsonl data/**/*.json

# Or remove specific files:
git rm --cached data/judge/train.jsonl
git rm --cached data/judge/val.jsonl
# ... etc for all tracked data files

# Commit the removal
git commit -m "Remove data files from public repo - now in private repo"
```

## What's Tracked vs Private

### ✅ Tracked in This Repository (Public)
- `data/generators/` - Data generation scripts
- `data/wrappers/` - Data wrapper utilities
- `data/resources/` - Resource files (TypeScript, schemas)
- `data/scenarios/` - Example scenario templates
- `data/README.md` - Documentation
- `scripts/` - Generation and processing scripts
- `schemas/` - Data schemas and validation

### 🔒 Private (Not in This Repository)
- `data/*.jsonl` - All dataset files
- `data/*.json` - All JSON data files
- `data/kd_cache/` - Cached API responses
- `data/logits/` - Cached logits
- `data/checkpoints/` - Dataset checkpoints
- `data/_snapshot_*/` - Data snapshots
- `data/_backup_*/` - Data backups

## Verification

Check what's currently tracked:

```bash
# List tracked data files (should be empty or minimal)
git ls-files data/ | grep -E "\.(jsonl|json|ndjson)$"

# Verify files are ignored
git check-ignore -v data/kd_mix.jsonl
git check-ignore -v data/judge/train.jsonl
```

## Backup Strategy

For your private data repository:

1. **Regular commits** to track changes
2. **Tag releases** for important dataset versions
3. **External backup** (e.g., cloud storage, separate backup system)
4. **Multiple remotes** for redundancy

## See Also

- `.gitignore` - See data ignore patterns
- `data/README.md` - Data directory documentation
- `scripts/README.md` - Data generation scripts

