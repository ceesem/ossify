#!/bin/bash
# Push the release commit and its tag together.
#
# `git push --tags` pushes tags but not the branch, so the tag can land on the
# remote pointing at a commit that is not reachable from any branch there until
# someone remembers to push separately. `--follow-tags` pushes the current
# branch along with the annotated tags reachable from it, which is exactly the
# release commit and its version tag. bump-my-version writes annotated tags
# (it sets tag_message), so they qualify.
git push --follow-tags
