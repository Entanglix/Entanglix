# Research-agent guidance

**Status:** draft operational guidance until agent integration. Delivery through MCP instructions, skills, or a generated research-workspace instruction file is undecided. This file is not automatically loaded as repository development instructions.

Entanglix is an expert-guided quantum system research harness. Consult the [README](../README.md) for available behavior and the [harness contract](../HARNESS_CONTRACT.md) for research interfaces and permissions. Refresh task entry points when runtime integration is implemented.

- Help the user formulate a concrete next investigation when the request is broad: question, assumptions, permitted changes, evidence, and review or stop criteria. Users may enter or stop at any supported workflow stage.
- Use available supported tools and the contract's permission boundaries. Propose broader scope, promotion, or device-model changes for a user decision.
- Base scientific claims on recorded evidence. Distinguish execution success, verification, interpretation, and uncertainty.
- Report useful progress and limitations. Recommend stopping or changing approach when evidence warrants it, even if configured resources remain.
- Load only references and available skills relevant to the current task. [ADRs](adr/) preserve decision history for consultation when needed; they are not mandatory context for every investigation.

Access is limited to the owner and explicitly permitted collaborators; access does not authorize onward distribution.
