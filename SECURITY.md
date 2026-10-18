# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting
instead: **Security → Report a vulnerability** on this repository. You can expect a first
reply within a week.

## Scope

WorldBuilder is meant to run on your own machine. Things worth knowing:

- The web app (`webapp/server.py`) has no login. It listens on `127.0.0.1` unless started
  with `--host 0.0.0.0` or `WORLDBUILDER_HOST`; anyone who can reach it can upload photos and
  start GPU jobs. Only expose it on a network you trust.
- The Blender, Unreal and MuJoCo integrations download scenes from the server you point them
  at. Point them only at servers you run or trust.
- Model weights are downloaded from Hugging Face under their own licences and are never
  distributed by this project.

Only the latest release receives fixes.
