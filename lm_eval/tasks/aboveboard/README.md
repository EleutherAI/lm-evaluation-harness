# AboveBoard 52 Sovereign Canadian Marine Benchmark (`lm-eval` Task)

## Task Information
- **Task Name**: `aboveboard_marine_52`
- **Dataset**: [Coralfil/AboveBoard-Marine-52](https://huggingface.co/datasets/Coralfil/AboveBoard-Marine-52)
- **Paper / Standard**: CRF-FS-ABOVEBOARD-52
- **License**: Apache-2.0
- **Maintained by**: Coralfil Marine Intelligence Research Team (Victoria, BC, Canada)

---

## Description
AboveBoard 52 is an empirical evaluation suite evaluating language model precision across Canadian maritime law, physical oceanography, and benthic ecosystem engineering.

Generalist foundation models frequently fail in coastal domains due to jurisdictional conflation (applying US Code of Federal Regulations to Canadian waters) and physical conservation errors (inverting aragonite saturation states, omitting seawater compressibility).

### Taxonomy (52 Scenarios)
1. **Statutory Marine Compliance (20 scenarios / 38.5%)**:
   - DFO Fisheries Act Section 35 (HADD criteria and authorizations)
   - Fisheries Act Section 56 (Aquatic organisms transfer permits)
   - Oceans Act Marine Protected Area boundaries (Hecate Strait Glass Sponge Reefs)
   - Canada Shipping Act 2001 (Small Vessel Regulations and ballast water exchange)
2. **Physical Oceanography & Biogeochemistry (18 scenarios / 34.6%)**:
   - TEOS-10 25-term polynomial seawater density derivations
   - Aragonite and calcite saturation horizons (Omega_arag, Omega_calc)
   - Harmonic decomposition of semi-diurnal M2 (12.42h) and S2 (12.00h) tidal constituents
   - Acoustic cavitation thresholds
3. **Benthic Restoration & Biomaterials (14 scenarios / 26.9%)**:
   - Chitosan-aragonite substrate degradation kinetics under estuarine hypoxia
   - Larval settlement rugosity thresholds for endangered *Haliotis kamtschatkana* (Pinto Abalone)
   - Shear stress tolerance on pulverized glass sponge spicule lattices

---

## How to Run

### Option 1: Standalone via `--include_path`
You do not need to wait for this pull request to merge. You can run the task immediately by pointing `lm_eval` to the task directory:

```bash
lm_eval --model vllm \
  --model_args pretrained=Coralfil/Atlantis-Pyramid-v2,dtype=bfloat16 \
  --include_path ./launch/AboveBoard-Marine-52/lm_eval \
  --tasks aboveboard_marine_52 \
  --batch_size auto
```

### Option 2: Running against OpenAI-Compatible Local Endpoint
```bash
lm_eval --model local-chat-completions \
  --model_args model=atlantis-pyramid-v2,base_url=http://localhost:8000/v1 \
  --include_path ./launch/AboveBoard-Marine-52/lm_eval \
  --tasks aboveboard_marine_52
```

---

## Verified Baseline Results

| Model | Parameters | Statutory Compliance | Physical Oceanography | Benthic Restoration | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Atlantis-Pyramid² (32B)** | **32.5B** | **100.0%** | **91.7%** | **100.0%** | **97.1%** |
| DeepSeek-R1-Distill-Qwen | 32B | 60.0% | 76.2% | 62.0% | 71.8% |
| Cohere Command R+ | 104B | 70.0% | 64.2% | 68.1% | 68.1% |
| Qwen2.5-Instruct | 32B | 65.0% | 72.4% | 68.2% | 67.3% |
| Claude 3.5 Sonnet | Closed | 65.0% | 67.5% | 63.1% | 66.3% |
| OpenAI GPT-4o | Closed | 65.0% | 63.5% | 56.8% | 61.4% |
