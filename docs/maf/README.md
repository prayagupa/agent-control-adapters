# Microsoft Agent Framework: ACS controls

The `agent-control-maf` alpha governs an existing native `agent_framework.Agent`
through the framework's experimental
`create_agent_hooks_middleware_from_emitter` factory. It does not provide a second
policy evaluator or message mapper. Only ACS policies are accepted.

- [Install and run](./installation.md)
- [Supported profile, enforcement, and limits](./coverage.md)
- [Complete local-Qwen examples](./examples.md)

Use `with_agent_controls(backend=Backend.MAF, runtime=agent, policy_file=...)`
and `await adapter.arun(RunRequest({"prompt": "..."}))` on the application's
existing event loop. Native client and Agent construction remain application-owned.
