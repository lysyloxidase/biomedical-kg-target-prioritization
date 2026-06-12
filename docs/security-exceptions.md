# Security Exceptions

## CVE-2025-3000

Status: temporarily ignored by `pip-audit`.

Reviewed: June 12, 2026.

Rationale:

- Public CVE and NVD descriptions identify PyTorch `2.6.0` and
  `torch.jit.script`.
- The lockfile currently resolves PyTorch `2.12.0`.
- Repository code does not invoke `torch.jit` or `torch.jit.script`.
- The PyPI advisory range currently covers all versions and declares no fixed
  release, so dependency resolution cannot clear the finding.

CI uses `--ignore-vuln CVE-2025-3000` and no broader vulnerability exclusion.

GitHub Dependency Review is additionally available when the repository has
Dependency Graph enabled and the repository variable
`ENABLE_DEPENDENCY_REVIEW=true`. The mandatory `pip-audit` and Bandit jobs do
not depend on this optional GitHub repository feature.

This is risk acceptance, not proof that PyTorch is vulnerability-free. Review
the exception whenever the lockfile or upstream advisory changes.
