# FLORES-200

### Paper

Title: `SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects`

Link: https://arxiv.org/abs/2309.07445

Despite the progress we have recorded in the last few years in multilingual natural language processing, evaluation is typically limited to a small set of languages with available datasets which excludes a large number of low-resource languages. In this paper, we created SIB-200 -- a large-scale open-sourced benchmark dataset for topic classification in 200 languages and dialects to address the lack of evaluation dataset for Natural Language Understanding (NLU). For many of the languages covered in SIB-200, this is the first publicly available evaluation dataset for NLU. The dataset is based on Flores-200 machine translation corpus. We annotated the English portion of the dataset and extended the sentence-level annotation to the remaining 203 languages covered in the corpus. Despite the simplicity of this task, our evaluation in full-supervised setting, cross-lingual transfer setting and prompting of large language model setting show that there is still a large gap between the performance of high-resource and low-resource languages when multilingual evaluation is scaled to numerous world languages. We found that languages unseen during the pre-training of multilingual language models, under-represented language families (like Nilotic and Altantic-Congo), and languages from the regions of Africa, Americas, Oceania and South East Asia, often have the lowest performance on our topic classification dataset. We hope our dataset will encourage a more inclusive evaluation of multilingual language models on a more diverse set of languages.

HuggingFace Page: https://huggingface.co/datasets/Davlan/sib200

We use the prompt template introduced by "MaLA-500: Massive Language Adaptation of Large Language Models" https://arxiv.org/abs/2401.13303v1, and then further used in "SambaLingo: Teaching Large Language Models New Languages" https://arxiv.org/abs/2404.05829.

Prompt template 
```
Topic Classification: science/technology,
travel, politics, sports, health, entertainment, geography.
The label of [sent] is [label]
```

### Citation

```
@misc{adelani2024sib200,
      title={SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for Topic Classification in 200+ Languages and Dialects}, 
      author={David Ifeoluwa Adelani and Hannah Liu and Xiaoyu Shen and Nikita Vassilyev and Jesujoba O. Alabi and Yanke Mao and Haonan Gao and Annie En-Shiun Lee},
      year={2024},
      eprint={2309.07445},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Tasks
* sib200_ckb_Arab
* sib200_kab_Latn
* sib200_mni_Beng
* sib200_srp_Cyrl
* sib200_crh_Latn
* sib200_kac_Latn
* sib200_mos_Latn
* sib200_ssw_Latn
* sib200_cym_Latn
* sib200_kam_Latn
* sib200_mri_Latn
* sib200_sun_Latn
* sib200_ace_Arab
* sib200_dan_Latn
* sib200_kan_Knda
* sib200_mya_Mymr
* sib200_swe_Latn
* sib200_ace_Latn
* sib200_deu_Latn
* sib200_kas_Arab
* sib200_nld_Latn
* sib200_swh_Latn
* sib200_acm_Arab
* sib200_dik_Latn
* sib200_kas_Deva
* sib200_nno_Latn
* sib200_szl_Latn
* sib200_acq_Arab
* sib200_dyu_Latn
* sib200_kat_Geor
* sib200_nob_Latn
* sib200_tam_Taml
* sib200_aeb_Arab
* sib200_dzo_Tibt
* sib200_kaz_Cyrl
* sib200_npi_Deva
* sib200_taq_Latn
* sib200_afr_Latn
* sib200_ell_Grek
* sib200_kbp_Latn
* sib200_nqo_Nkoo
* sib200_taq_Tfng
* sib200_ajp_Arab
* sib200_eng_Latn
* sib200_kea_Latn
* sib200_nso_Latn
* sib200_tat_Cyrl
* sib200_aka_Latn
* sib200_epo_Latn
* sib200_khk_Cyrl
* sib200_nus_Latn
* sib200_tel_Telu
* sib200_als_Latn
* sib200_est_Latn
* sib200_khm_Khmr
* sib200_nya_Latn
* sib200_tgk_Cyrl
* sib200_amh_Ethi
* sib200_eus_Latn
* sib200_kik_Latn
* sib200_oci_Latn
* sib200_tgl_Latn
* sib200_apc_Arab
* sib200_ewe_Latn
* sib200_kin_Latn
* sib200_ory_Orya
* sib200_tha_Thai
* sib200_arb_Arab
* sib200_fao_Latn
* sib200_kir_Cyrl
* sib200_pag_Latn
* sib200_tir_Ethi
* sib200_arb_Latn
* sib200_fij_Latn
* sib200_kmb_Latn
* sib200_pan_Guru
* sib200_tpi_Latn
* sib200_ars_Arab
* sib200_fin_Latn
* sib200_kmr_Latn
* sib200_pap_Latn
* sib200_tsn_Latn
* sib200_ary_Arab
* sib200_fon_Latn
* sib200_knc_Arab
* sib200_pbt_Arab
* sib200_tso_Latn
* sib200_arz_Arab
* sib200_fra_Latn
* sib200_knc_Latn
* sib200_pes_Arab
* sib200_tuk_Latn
* sib200_asm_Beng
* sib200_fur_Latn
* sib200_kon_Latn
* sib200_plt_Latn
* sib200_tum_Latn
* sib200_ast_Latn
* sib200_fuv_Latn
* sib200_kor_Hang
* sib200_pol_Latn
* sib200_tur_Latn
* sib200_awa_Deva
* sib200_gaz_Latn
* sib200_lao_Laoo
* sib200_por_Latn
* sib200_twi_Latn
* sib200_ayr_Latn
* sib200_gla_Latn
* sib200_lij_Latn
* sib200_prs_Arab
* sib200_tzm_Tfng
* sib200_azb_Arab
* sib200_gle_Latn
* sib200_lim_Latn
* sib200_quy_Latn
* sib200_uig_Arab
* sib200_azj_Latn
* sib200_glg_Latn
* sib200_lin_Latn
* sib200_ron_Latn
* sib200_ukr_Cyrl
* sib200_bak_Cyrl
* sib200_grn_Latn
* sib200_lit_Latn
* sib200_run_Latn
* sib200_umb_Latn
* sib200_bam_Latn
* sib200_guj_Gujr
* sib200_lmo_Latn
* sib200_rus_Cyrl
* sib200_urd_Arab
* sib200_ban_Latn
* sib200_hat_Latn
* sib200_ltg_Latn
* sib200_sag_Latn
* sib200_uzn_Latn
* sib200_bel_Cyrl
* sib200_hau_Latn
* sib200_ltz_Latn
* sib200_san_Deva
* sib200_vec_Latn
* sib200_bem_Latn
* sib200_heb_Hebr
* sib200_lua_Latn
* sib200_sat_Olck
* sib200_vie_Latn
* sib200_ben_Beng
* sib200_hin_Deva
* sib200_lug_Latn
* sib200_scn_Latn
* sib200_war_Latn
* sib200_bho_Deva
* sib200_hne_Deva
* sib200_luo_Latn
* sib200_shn_Mymr
* sib200_wol_Latn
* sib200_bjn_Arab
* sib200_hrv_Latn
* sib200_lus_Latn
* sib200_sin_Sinh
* sib200_xho_Latn
* sib200_bjn_Latn
* sib200_hun_Latn
* sib200_lvs_Latn
* sib200_slk_Latn
* sib200_ydd_Hebr
* sib200_bod_Tibt
* sib200_hye_Armn
* sib200_mag_Deva
* sib200_slv_Latn
* sib200_yor_Latn
* sib200_bos_Latn
* sib200_ibo_Latn
* sib200_mai_Deva
* sib200_smo_Latn
* sib200_yue_Hant
* sib200_bug_Latn
* sib200_ilo_Latn
* sib200_mal_Mlym
* sib200_sna_Latn
* sib200_zho_Hans
* sib200_bul_Cyrl
* sib200_ind_Latn
* sib200_mar_Deva
* sib200_snd_Arab
* sib200_zho_Hant
* sib200_cat_Latn
* sib200_isl_Latn
* sib200_min_Arab
* sib200_som_Latn
* sib200_zsm_Latn
* sib200_ceb_Latn
* sib200_ita_Latn
* sib200_min_Latn
* sib200_sot_Latn
* sib200_zul_Latn
* sib200_ces_Latn
* sib200_jav_Latn
* sib200_mkd_Cyrl
* sib200_spa_Latn
* sib200_cjk_Latn
* sib200_jpn_Jpan
* sib200_mlt_Latn
* sib200_srd_Latn


### Checklist

For adding novel benchmarks/datasets to the library:
  * [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
