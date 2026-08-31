# Contributing to rl-insight

Thank you for considering a contribution to rl-insight! We welcome contributions of any kind - bug fixes, enhancements, documentation improvements, or even just feedback. Whether you're an experienced developer or this is your first open-source project, your help is invaluable.

## Community

Join our Lark group to connect with the team and other developers:

<p align="center">
  <img src="https://raw.githubusercontent.com/verl-project/rl-insight/main/assets/community/lark-qr.png" width="240" alt="Join RL-Insight on Lark">
</p>

Your support can take many forms:
- Report issues or unexpected behaviors.
- Suggest or implement new features.
- Improve or expand documentation.
- Review pull requests and assist other contributors.
- Spread the word: share rl-insight in blog posts, social media, or give the repo a ⭐.

## Finding Issues to Contribute

Looking for ways to dive in? Check out these issues:
- [Good first issues](https://github.com/verl-project/rl-insight/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
- [Call for contribution](https://github.com/verl-project/rl-insight/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22call%20for%20contribution%22)
Furthermore, you can learn the development plan and roadmap via [RFC](https://github.com/verl-project/rl-insight/issues?q=is%3Aissue%20state%3Aopen%20label%3ARFC) and [Roadmap](https://github.com/verl-project/rl-insight/issues?q=state%3Aopen%20label%3A%22roadmap%22).

## Developing

- Create and activate a Python virtual environment (Python >= 3.9).
- Install the project in editable mode:

```bash
pip install -e ".[test]"
```

When developing or testing Recipe, include its optional dependencies:

```bash
pip install -e ".[recipe,test]"
```

## Code Linting and Formatting

We rely on pre-commit to keep our code consistent. To set it up:

```bash
pip install pre-commit
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
# run a specific hook (example)
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always mypy
```

## Testing

Run the test suite locally:

```bash
pytest
```

### Adding CI tests

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow file under `.github/workflows/`.
2. Add related path patterns to the `paths` section if not already included.
3. Minimize the workload of the test script(s) (see existing workflows for examples).

## Building the Docs

Currently the documentation is maintained as markdown files under the `docs/` directory.
You can preview them locally with any markdown viewer or render them using your preferred static site generator.

## Pull Requests & Code Reviews

Thanks for submitting a PR! To streamline reviews:
- Follow our Pull Request Template for title format and checklist.
- Adhere to our pre-commit lint rules and ensure all checks pass.
- Update docs for any user-facing changes.
- Add or update tests in the CI workflows, or explain why tests aren't applicable.

## License

See the [LICENSE](https://github.com/verl-project/rl-insight/blob/main/LICENSE) file for full details.

## Thank You

We appreciate your contributions to rl-insight. Your efforts help make the project stronger and more user-friendly. Happy coding!
