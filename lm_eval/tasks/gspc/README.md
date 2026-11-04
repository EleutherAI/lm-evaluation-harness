# GSPC — Governance/Safety/Provenance/Continuity banks (Council of AI)

Six frozen public banks from the Council of AI (CSOAI Ltd) GSPC estate, exposed as
lm-eval multiple-choice tasks for external-harness reproduction.

### Citation

```bibtex
@misc{csoai2026gspc,
  title={GSPC: Governance, Safety, Provenance and Continuity measurement estate},
  author={CSOAI Ltd},
  year={2026},
  url={https://councilof.ai}
}
```

### Tasks
`gspc_mcp` · `gspc_oss` · `gspc_prv` · `gspc_asi` · `gspc_det` · `gspc_art5`

- Banks live at `csoai/gspc-*` on the Hub (CC-BY-4.0, canaried to prevent training contamination — canary rows are filtered by each task's `process_docs`).
- Banks name their content column differently (`tool`/`item`/`operation`/`case`) — each YAML reads its own field; filters match per bank (`utils.py`).
- Metric: `acc` / `acc_norm` over the bank's frozen label set (exact-match lane grading).
- **Hub listing is not a GSPC grade** — canonical measurement stays at councilof.ai (board_v2 grader, signed cards). Empty cells stay empty; never certify; never stamp MEASURED from harness output.

### Task YAML

Each YAML carries the bank's honesty block verbatim (printer_not_mill, hub_listing_is_not_a_grade, never_llm_judge, never_stamp_measured, never_certify).

Author: CSOAI-ORG, 2026-11-04.
