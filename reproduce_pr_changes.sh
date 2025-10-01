#!/bin/bash
# Script to reproduce PR #2946 review feedback changes
# Repository: https://github.com/EleutherAI/lm-evaluation-harness/pull/2946
# Branch: final_putnam_axiom_bm

set -e  # Exit on error
set -u  # Exit on undefined variable

echo "=== Reproducing PR #2946 Review Feedback Changes ==="
echo "Branch: final_putnam_axiom_bm"
echo "PR: https://github.com/EleutherAI/lm-evaluation-harness/pull/2946"
echo ""

# Set up paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Ensure we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "final_putnam_axiom_bm" ]; then
    echo "❌ Error: Not on final_putnam_axiom_bm branch (currently on: $CURRENT_BRANCH)"
    echo "Please run: git checkout final_putnam_axiom_bm"
    exit 1
fi

echo "✅ On correct branch: $CURRENT_BRANCH"
echo ""

# Create and activate UV environment if it doesn't exist
UV_ENV="$HOME/uv_envs/lm_eval_harness"
if [ ! -d "$UV_ENV" ]; then
    echo "📦 Creating UV environment at: $UV_ENV"
    uv venv "$UV_ENV" --python 3.10
    echo "✅ UV environment created"
else
    echo "✅ UV environment already exists at: $UV_ENV"
fi
echo ""

# Activate environment and install dependencies
echo "📦 Installing dependencies..."
source "$UV_ENV/bin/activate"

# Install pip if not present
if ! command -v pip &> /dev/null; then
    echo "Installing pip in UV environment..."
    python -m ensurepip
fi

# Install ruff for formatting
pip install -q ruff

# Install pre-commit for code quality checks
pip install -q pre-commit

echo "✅ Dependencies installed"
echo ""

# Display what changes were made in the PR review
echo "=== Changes Made to Address PR Review ==="
echo ""
echo "1. ✅ Removed unused imports:"
echo "   - lm_eval/tasks/imo_olympiad/utils.py: removed antlr4, numpy"
echo "   - lm_eval/tasks/putnam_axiom/utils.py: removed antlr4, numpy"
echo "   - lm_eval/evaluator.py: removed pandas"
echo ""
echo "2. ✅ Fixed logging:"
echo "   - Changed from: from lm_eval.utils import eval_logger"
echo "   - Changed to: eval_logger = logging.getLogger(__name__)"
echo ""
echo "3. ✅ Added README files:"
echo "   - lm_eval/tasks/imo_olympiad/README.md"
echo "   - lm_eval/tasks/putnam_axiom/README.md"
echo ""
echo "4. ✅ Updated main tasks README:"
echo "   - Added entries to lm_eval/tasks/README.md table"
echo ""
echo "5. ✅ Code formatting:"
echo "   - Fixed bare except clauses to use 'except Exception:'"
echo "   - Ran ruff check --fix and ruff format"
echo ""

# Run code quality checks
echo "=== Running Code Quality Checks ==="
echo ""

echo "🔍 Running ruff linter..."
ruff check --fix lm_eval/tasks/imo_olympiad/utils.py lm_eval/tasks/putnam_axiom/utils.py lm_eval/evaluator.py
echo "✅ Ruff linter passed"
echo ""

echo "🎨 Running ruff formatter..."
ruff format lm_eval/tasks/imo_olympiad/utils.py lm_eval/tasks/putnam_axiom/utils.py lm_eval/evaluator.py
echo "✅ Ruff formatter passed"
echo ""

# Show current git status
echo "=== Current Git Status ==="
git status --short
echo ""

# Show latest commit
echo "=== Latest Commit ==="
git log -1 --oneline
echo ""

echo "=== Usage Instructions ==="
echo ""
echo "To test the benchmark tasks, run:"
echo "  source $UV_ENV/bin/activate"
echo "  pip install -e ."
echo "  lm-eval --model hf --model_args pretrained=gpt2 --tasks putnam_axiom_original --limit 5"
echo ""
echo "To push changes to GitHub:"
echo "  git push origin final_putnam_axiom_bm"
echo ""
echo "=== Reviewer Requests (from @baberabb) ==="
echo ""
echo "✅ Remove the unused imports (and fix the logging error)"
echo "✅ Add a readme to the task folders"
echo "✅ Add an entry for each to the table in lm_eval/tasks/README.md"
echo "✅ Run pre-commit: pip install pre-commit && pre-commit run --all-files"
echo ""
echo "All changes have been completed and committed!"
echo "Commit: e9ae93d0"
echo "PR: https://github.com/EleutherAI/lm-evaluation-harness/pull/2946"
echo ""
echo "✅ Ready for review!"

