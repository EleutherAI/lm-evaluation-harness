# GSPC

Eleven measurement tasks from the GSPC banks published by Council of AI (CSOAI Ltd). Each bank is a public dataset on the Hugging Face Hub.

| Task | What it asks | Dataset | Items |
|---|---|---|---|
| `gspc_care` | Crisis refusal and safeguarding of vulnerable people | `csoai/gspc-care` | 200 |
| `gspc_safety` | Operational AI safety refusal | `csoai/gspc-agi` | 36 |
| `gspc_detector_interop` | Provenance-detector consistency (C2PA-style) | `csoai/gspc-det` | 33 |
| `gspc_affect` | Emotional manipulation, sycophancy and dependency | `csoai/gspc-affect` | 41 |
| `gspc_provenance` | Cryptographic provenance and C2PA integrity | `csoai/gspc-prv` | 32 |
| `gspc_cross_reality` | Synthetic-media and spatial boundary cases | `csoai/gspc-xr` | 32 |
| `gspc_machinery` | Classification under the EU Machinery Regulation 2023/1230 | `csoai/gspc-mach` | 33 |
| `gspc_openness` | Model openness and weight and data provenance | `csoai/gspc-oss` | 32 |
| `gspc_continuity` | Post-quantum cryptographic agility | `csoai/gspc-asi` | 33 |
| `gspc_conformance` | MCP and A2A protocol conformance | `csoai/gspc-mcp` | 35 |
| `gspc_art5_safeguard` | EU AI Act Article 5 prohibited practices | `csoai/gspc-art5` | 36 |

- `--tasks gspc` runs all eleven through the `gspc` tag. Scores are reported per task; there is no aggregate score.
- Each bank holds one canary row, marked in its `_canary` field. `utils.process_docs` removes it, so it is never scored.
- Each bank's licence is set on its Hugging Face dataset card.
- The signed measurement board these banks feed: https://councilof.ai/api/gspc
