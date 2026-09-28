# Preparing a release

The `Verified release` workflow prepares a GitHub draft release from a version tag. It does not publish the release or upload anything to PyPI.

## Maintainer steps

1. Merge the release-ready changes to `main` and set the version in `pyproject.toml`.
2. Create and push an annotated tag whose name is `v` followed by that exact version. For example, version `0.1.0` uses tag `v0.1.0`.
3. Wait for the `Verified release` workflow to finish. It runs the tests, lint and type checks, builds and smoke-tests the wheel and source distribution, checks the tag against the package version, calculates SHA-256 checksums, attests the distributions, verifies their checksums and attestations, and creates a draft GitHub release with the distributions and checksum file attached.
4. Review the generated release notes and attached files in GitHub. Publish the draft manually when they are ready.

The workflow rejects a tag if it does not match the version in `pyproject.toml`. It also requires the tag to exist on GitHub before creating the release. The workflow definition must be present in the tagged commit.

Consumers can verify a downloaded distribution's provenance with GitHub CLI:

```sh
gh attestation verify llmigrate-0.1.0-py3-none-any.whl --repo ilya-kolchinsky/llmigrate
```

Use the actual wheel filename for the release being verified. Verify the source distribution the same way. The `SHA256SUMS` asset lets consumers check that a downloaded file matches the published release asset.
