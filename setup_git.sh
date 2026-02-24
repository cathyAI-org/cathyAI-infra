#!/bin/bash
# Git setup script for cathyAI-infra development workflow

echo "Setting up git environment for cathyAI-infra..."

# Fetch all branches
git fetch origin

# Check if dmz branch exists locally
if git show-ref --verify --quiet refs/heads/dmz; then
    echo "✓ dmz branch already exists locally"
else
    echo "Creating local dmz branch..."
    git checkout -b dmz origin/dmz
fi

# Switch to dmz branch
git checkout dmz

# Set dmz as upstream
git branch --set-upstream-to=origin/dmz dmz

echo "✓ Git environment configured!"
echo "You are now on the 'dmz' branch"
echo "Push with: git push"
echo ""
echo "Setting up commit message template..."
git config commit.template .gitmessage
echo "✓ Commit template configured"
