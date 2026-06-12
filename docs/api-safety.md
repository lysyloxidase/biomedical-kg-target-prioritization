# API Safety

## Fail-Closed Readiness

The normal API validates a completed run manifest plus checkpoint, model
configuration, dataset manifest, graph manifest, feature manifest, node-index
map, split metadata, validation metrics, and supervision/message-edge
separation.

Missing, corrupt, untrained, smoke-only, or incompatible artifacts cause HTTP
503. Container health uses `/health`, so an image without validated mounted
artifacts is not marked ready.

## Prediction Metadata

Responses include run ID, model name and version, checkpoint SHA-256, dataset
ID and manifest SHA-256, split ID, trained status, validation metrics, candidate
protocol, warnings, and `hypothesis_only: true`.

## Demo Mode

`KGTP_DEMO_MODE=true` explicitly enables a one-epoch model on a synthetic
smoke graph. Demo responses are marked non-scientific. There is no implicit
fallback from production mode to demo mode.

## Operational Limitations

Authentication, authorization, rate limiting, TLS termination, request
timeouts, audit logging, and production monitoring are deployment
responsibilities and are not implemented by this research API.

PyTorch checkpoints use pickle-based serialization and must come from a trusted
source.
