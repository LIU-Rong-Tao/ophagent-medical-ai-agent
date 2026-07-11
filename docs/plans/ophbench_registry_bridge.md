# Ophbench registry bridge implementation plan

## Goal

Allow OphAgent to consume the external `ophbench` registry as an optional, metadata-only model
catalog without copying registry files, downloading weights, or changing current routing behavior.

## Boundaries

- Add one independent bridge module and tests; do not edit current Model Hub routing/UI modules.
- Use `ophbench.registry.loader.load_registry` as the data API.
- Treat upstream foundation checkpoints as non-task checkpoints.
- Return structured dependency and registry errors instead of failing application startup.
- Record upstream package version, Git commit, registry root, and UTC load time.
- Verify the production Python 3.10 environment degrades cleanly and use an isolated Python 3.11
  environment for the real editable-install integration test.

## TDD tasks

1. Add failing unit tests for status mapping, RETFound checkpoint/authentication metadata,
   dependency absence, registry errors, and task-checkpoint separation.
2. Implement the smallest pure-Python bridge API that passes those tests.
3. Add a separately runnable integration test for the real 15-model/27-checkpoint registry.
4. Run the existing OphAgent test suite and `compileall` under the production environment.
5. Run the real bridge integration under an isolated Python 3.11 venv with editable ophbench.
6. Commit and push `feat/ophbench-registry-bridge`; do not merge, tag, or release.
