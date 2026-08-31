# Security policy

I want security reports for `fast-attnres` to reach me before sensitive
details reach the public issue tracker. This repository contains training,
evaluation, and benchmark code; it may be run with credentials for model hubs,
experiment trackers, or dataset services, so treat scripts and logs as
potentially sensitive.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting or a private Security
Advisory for this repository when that feature is available. If private
reporting is not enabled, contact the maintainer through the maintainer's
GitHub profile and ask for a private reporting channel. Do not publish exploit
steps, credentials, access tokens, private datasets, or personal data in a
public issue.

Include enough information to reproduce the problem safely:

- affected commit, file, and configuration;
- the impact and the trust boundary that is crossed;
- a minimal reproduction or safe proof of concept;
- whether exploitation requires a local account, a service credential, or
  another precondition; and
- any mitigation you have already applied.

Please allow time for triage and remediation before making a report public. I
will keep the report private while I investigate and will coordinate public
disclosure when appropriate.

## Credentials and experiment data

Never paste a real secret into an issue, pull request, benchmark artifact, or
chat transcript. Use environment variables or a local credential store, and
replace any accidental disclosure with `[REDACTED_SECRET]` in shared material.
If a token or key may have been exposed, revoke or rotate it immediately; do
not rely on deleting the message or file.

Before uploading logs, checkpoints, or benchmark reports, remove credentials,
private prompts, private datasets, user identifiers, and local paths that
reveal sensitive information. Artifacts should contain only the metadata needed
to reproduce the stated result.

## Supported versions

Security fixes are considered for the latest code on the default branch. Older
commits and experimental branches may not receive a backport. If you are
running a released or archived snapshot, include its exact commit when
reporting an issue.

## Scope

This policy covers the code and documentation in this repository, including
data preparation, training, evaluation, source-list kernels, and benchmark
tooling. It does not make a security guarantee about third-party dependencies,
downloaded datasets, model checkpoints, hosted services, or infrastructure
outside this repository; report those issues to the relevant provider as well.
