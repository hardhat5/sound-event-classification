# Contributing

An average contribution would involve the following:

1. Fork this repository in your account.
2. Clone it on your local machine.
3. Add a new remote using `git remote add upstream https://github.com/hardhat5/sound-event-classification.git`.
4. Create a new feature branch with `git checkout -b my-feature`.
5. Make your changes.
6. Commit your changes.
7. Rebase your commits with `upstream/main`:
  - `git checkout main`
  - `git fetch upstream main`
  - `git reset --hard FETCH_HEAD`
  - `git checkout my-feature`
  - `git rebase main
8. Resolve any merge conflicts, and then push the branch with `git push origin my-feature`.
9. Create a Pull Request detailing the changes you made and wait for review/merge.
