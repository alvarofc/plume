# Security Policy

## Reporting a vulnerability

Please do not open public GitHub issues for suspected security vulnerabilities.

Instead, report them privately to the maintainer with:

- A description of the issue
- Affected versions or commit range
- Reproduction steps or proof of concept
- Any suggested mitigation

If you already have a private disclosure address or GitHub security advisory workflow configured, use that channel. Otherwise, establish one before publishing this repository broadly.

## Scope

Security-sensitive areas in this project include:

- HTTP request handling in `plume-api`
- Filesystem and object-store access
- Release/install pipeline (`install.sh`, GitHub Actions, release artifacts)
- Dependency and model download paths

## Supported versions

Until versioned support windows are defined, treat the latest `main` branch and the most recent tagged release as supported for security fixes.
