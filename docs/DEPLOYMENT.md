# Deployment outline

## Local Compose smoke topology

`compose.yml` starts a Flower 1.31 SuperLink, four SuperNodes with distinct partition IDs, a ServerApp runtime, four ClientApp runtimes, and a synthetic-data initializer. The smoke dataset is generated once and mounted read-only into each client runtime. Start the topology with `docker compose up --build -d`, then launch one run from the host with `scripts/run_compose_app.sh`; stop it with `docker compose down`.

This profile uses Flower's insecure transport for local development. All four client containers share one machine and one synthetic dataset volume. It is a containerized smoke test, not a multi-organization deployment or evidence of authenticated client identity. Use TLS and separately registered SuperNode keys for a real federation; see `docs/DECISIONS.md` D-STAGE-F-001.

## Provider-neutral production layout

- Run the SuperLink control plane in a dedicated, access-controlled network segment with durable state storage and monitored backups.
- Run one SuperNode and its ClientApp runtime in each participating organization's environment. Keep each site's data and credentials inside that site; exchange only the protocol messages needed by the federated workflow.
- Use TLS with an organization-managed CA, register SuperNode public keys, rotate credentials through a secret manager, and restrict inbound and outbound ports to the required Flower APIs.
- Pin container images by release and digest, scan the image and Python dependency lock, apply resource limits, and retain build provenance and audit records.
- Define recovery, retention, incident response, availability, and cross-region requirements with the participating organizations before deployment.

The repository has not been deployed to a cloud provider. Network boundaries, identity, availability, and governance require validation in the chosen organizations' environments.
