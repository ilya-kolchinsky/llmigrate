# Preparing a release

The `Verified release` workflow prepares a GitHub draft release from a version tag. After you publish that draft, the `Publish to PyPI` workflow verifies the attached distributions and uploads them to PyPI using Trusted Publishing.

## Maintainer steps

1. Merge the release-ready changes to `main` and set the version in `pyproject.toml`.
2. Create and push an annotated tag whose name is `v` followed by that exact version. For example, version `0.1.0` uses tag `v0.1.0`.
3. Wait for the `Verified release` workflow to finish. It runs the tests, lint and type checks, builds and smoke-tests the wheel and source distribution, checks the tag against the package version, calculates SHA-256 checksums, attests the distributions, verifies their checksums and attestations, and creates a draft GitHub release with the distributions and checksum file attached.
4. Review the generated release notes and attached files in GitHub, then publish the draft. The `Publish to PyPI` workflow downloads those same release files, checks their SHA-256 checksums and provenance attestations, and publishes the wheel and source distribution to PyPI.

## One-time PyPI setup

Before the first upload:

1. Check that the `llmigrate` project name is available on PyPI. A pending publisher does not reserve the name until the first upload.
2. Create a GitHub Actions Trusted Publisher on PyPI for project `llmigrate`, repository owner `ilya-kolchinsky`, repository `llmigrate`, workflow file `publish.yml`, and GitHub Actions environment `pypi`.
3. Create a GitHub Actions environment named `pypi` in the repository. No PyPI API token or GitHub secret is required.

The current GitHub release `v0.1.0` was published before the PyPI workflow was added, so its release event will not trigger this workflow. After the workflow is merged to `main` and the publisher is configured, run `Publish to PyPI` manually from the Actions tab on `main`, providing `v0.1.0` as the tag. Future published GitHub releases trigger PyPI publishing automatically.

After publishing, verify the project page and install the release in a clean environment:

```sh
python -m pip install llmigrate==0.1.0
python -I -c "import llmigrate; assert callable(llmigrate.migrate)"
```

The workflow rejects a tag if it does not match the version in `pyproject.toml`. It also requires the tag to exist on GitHub before creating the release. The workflow definition must be present in the tagged commit.

Consumers can verify a downloaded distribution's provenance with GitHub CLI:

```sh
gh attestation verify llmigrate-0.1.0-py3-none-any.whl --repo ilya-kolchinsky/llmigrate
```

Use the actual wheel filename for the release being verified. Verify the source distribution the same way. The `SHA256SUMS` asset lets consumers check that a downloaded file matches the published release asset.
