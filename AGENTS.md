# Repository instructions

Keep changes focused, preserve untracked files and existing user work, and follow
the repository's established tooling. Do not assume Python, formatter, or CUDA
versions from a related project apply here; use this project's declared versions
and the supported ComfyUI environment.

## Privacy

Treat repository content, filenames, commit metadata and messages, PRs, issues,
comments, releases, and uploaded artifacts as potentially public.

- Use repository-relative references and commands in publication text and
  generated metadata. Do not include local absolute paths, home-directory
  shortcuts, machine usernames, hostnames, private network addresses, or other
  machine-identifying details.
- Keep credentials, tokens, private keys, personal email addresses, and private
  account details out of tracked files and outgoing text. Do not copy local
  environment files or credential-bearing configuration into the repository.
- Keep raw machine captures, logs, environment dumps, and benchmark captures
  outside the tracked tree. Review and sanitize any extracts before publication.
- Inspect author and committer metadata before committing. Never publish
  unreviewed shell output or attribution trailers, including `Co-authored-by`,
  `Reviewed-by`, or `Signed-off-by` trailers containing personal email addresses.
- If a privacy check finds a match, report a generic failure without echoing the
  sensitive content. Correct the source and check again; do not bypass guards.

## Worktree workflow

Once Git has an initial commit, create a task branch and dedicated worktree from
the current upstream base (or local HEAD when no upstream exists) before editing
files. Perform task edits, checks, commits,
and publication preparation inside that worktree. Keep the primary checkout
clean; do not edit it first and then copy changes into a worktree or leave
duplicate uncommitted task files there after publishing. Initial local bootstrap
work can happen in this directory until Git has an initial commit.

Periodically, especially before starting new work and after a PR merges, fetch
upstream and inspect `git worktree list`. Clean up completed task worktrees and
their local branches only after confirming the changes are merged upstream. For
squash or rebase merges, verify the PR state and resulting content rather than
relying only on commit ancestry.

Before removing a worktree, inspect its index, working files, and untracked files
and confirm the task is finished. Preserve unfinished or untracked work and leave
active or unrelated worktrees alone. Do not force-remove worktrees or use broad
cleanup commands that discard files. After merged work is cleaned up,
fast-forward the primary checkout when possible and verify that it is clean.

## Publication workflow

Commit and publish only within the user's authorized scope.

The privacy checker, hooks, and guard tests referenced below must be installed
and validated in this repository before the first commit or publication. This
instructions file alone does not install them. If a required check is unavailable
or cannot run, stop before committing or publishing; local editing may continue.

Install local hooks with `git config core.hooksPath .githooks` once the hooks
exist. Hooks supplement this workflow; do not bypass them.

Before each commit, inspect the complete staged diff and exact commit message,
save the message in a repository-relative temporary file, then run:

```sh
python3 tools/check_public_text.py --staged --metadata
python3 tools/check_public_text.py --text-file commit-message.txt
```

Commit with precisely the checked message. Keep temporary publication files
untracked and out of the staged changes.

Before each push, run:

```sh
python3 tools/check_public_text.py --history --metadata
```

Before creating or editing a PR, comment, issue, release, or other publication,
place the exact outgoing title and body in repository-relative temporary files
and scan each with `python3 tools/check_public_text.py --text-file FILE`. Send
precisely the checked text. Check again after any edit.

The guard must fail with a generic message without printing matching content.
Additional local aliases may be configured with repeated
`git config --add privacy.identity VALUE` entries; keep those values local.

## Validation

Run checks appropriate to the change using the project's declared tooling. When
the privacy guard is installed, run its tests with:

```sh
python3 -m unittest tests.test_public_text -v
```

For attention integration changes, verify tensor layouts, dtypes, masks, device
handling, and fallback behavior against the supported ComfyUI attention API.
Run relevant GPU checks for CUDA-dependent changes when suitable hardware is
available. State which checks ran and any hardware limitations; do not present
CPU-only checks as GPU validation.

For performance claims, record the GPU, software versions, tensor shapes,
dtypes, backend, and measurement method. Separate compilation and warm-up from
steady-state timings and distinguish attention-call measurements from
end-to-end ComfyUI performance. Sanitize the report before publication.
