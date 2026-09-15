# Security Policy

Kamino is in pre-release development and has no supported public version.
Report vulnerabilities privately to the repository owner when a remote exists.

The primary trust boundaries are formula parsing, untrusted result artifacts,
native numerical dependencies, resource-exhausting dimensions, and accidental
disclosure of caller data. The default runtime performs no network calls or
telemetry upload. Result loading will reject executable serialization formats.
