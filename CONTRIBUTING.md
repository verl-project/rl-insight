# Contributing to rl-insight

Thank you for considering a contribution to rl-insight! We welcome contributions of any kind - bug fixes, enhancements, documentation improvements, or even just feedback. Whether you're an experienced developer or this is your first open-source project, your help is invaluable.

Your support can take many forms:
- Report issues or unexpected behaviors.
- Suggest or implement new features.
- Improve or expand documentation.
- Review pull requests and assist other contributors.
- Spread the word: share rl-insight in blog posts, social media, or give the repo a ⭐.


## Developing

- Create and activate a Python virtual environment (Python >= 3.9).
- Install the project in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
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
# run a specific hook with pre-commit
# pre-commit run --all-files --show-diff-on-failure --color=always <hood-id>
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

## Testing

### Adding CI tests

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file, which usually corresponds to a `hydra` default config (e.g. `ppo_trainer`, `ppo_megatron_trainer`, `sft_trainer`, etc).
2. Add related path patterns to the `paths` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

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

