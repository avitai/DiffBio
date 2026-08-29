# Security Policy

## Supported Versions

DiffBio is a research preview. Security fixes target the latest released version
and the current `main` branch.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately by emailing
security@avitai.bio.

Include:

- Affected DiffBio version or commit.
- Environment details, including Python, JAX, and operating system versions.
- A minimal reproduction or proof of impact.
- Any known mitigations.

We will acknowledge reports as soon as practical, investigate privately, and
coordinate disclosure once a fix or mitigation is available.

## Scope

Security-sensitive areas include sequence, variant and molecular file parsing, dataset and reference-genome loading, checkpoint loading, file-system storage, CI release automation, and any code path that processes untrusted biological input data.
