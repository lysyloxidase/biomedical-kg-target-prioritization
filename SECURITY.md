# Security Policy

## Supported Version

Security fixes target the latest pre-1.0 development version.

## Reporting

Please report suspected vulnerabilities privately through GitHub Security
Advisories for this repository. Do not include credentials, private biomedical
data, or exploit details in public issues.

## Deployment Notes

- Never commit `.env` or credentials.
- Treat PyTorch checkpoints as trusted-code artifacts.
- Keep the API fail-closed and mount only validated read-only artifacts.
- Place public deployments behind TLS, authentication, rate limiting, and
  network controls.
- Neo4j is optional and requires an explicit password.

## Audited Exception

As of June 12, 2026, `pip-audit` maps `CVE-2025-3000` to all PyPI `torch`
versions and provides no fixed version. Public records describe `torch 2.6.0`
and `torch.jit.script`; this repository locks `torch 2.12.0` and does not call
the JIT scripting API. CI ignores only this CVE while continuing to fail on
every other reported vulnerability. The exception must be removed when
upstream advisory ranges or a fixed version become available.
