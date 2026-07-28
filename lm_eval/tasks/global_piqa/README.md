# Task-name

## Paper

Title: `Global PIQA`

Abstract: `To date, there exist almost no culturally-specific evaluation benchmarks for large language models (LLMs) that cover a large number of languages and cultures. We present Global PIQA, a participatory commonsense reasoning benchmark for over 100 languages, constructed by hand by 320 researchers from 65 countries around the world. The 116 language varieties in Global PIQA cover five continents, 14 language families, and 23 writing systems. In the non-parallel split of Global PIQA, over 50% of examples reference local foods, customs, traditions, or other culturally-specific elements. Beyond its uses for LLM evaluation, we hope that Global PIQA provides a glimpse into the wide diversity of cultures in which human language is embedded.`

Homepage: https://mrlbenchmarks.github.io/


You can run global_piqa in two formats: `global_piqa_generation` runs a generation-style evaluation for instruction-tuned and chat models. `global_piqa_cloze` runs log-likelihood or cloze-style evaluation, best for base models or small models.

### Groups, Tags, and Tasks

#### Groups

You can also run the parallel and non-parallel subsets individually in each format.

* `group_name`: `global_piqa_nonparallel_generation` Generation task using chat format
* `group_name`: `global_piqa_nonparallel_cloze` Cloze-style completion format
* `group_name`: `global_piqa_parallel_generation` Generation task using chat format
* `group_name`: `global_piqa_parallel_cloze` Cloze-style completion format

You can also run individual languages:
* Using the template `global_piqa_{split}_{format}_{iso_script}`, you can run individual language datasets, also by split (nonparallel or parallel) and format (generation or cloze). For example: `global_piqa_parallel_cloze_eng_latn`.

<details>

<summary>Languages available</summary>

**Nonparallel (136)**

    - acm_arab
    - acq_arab
    - aeb_arab
    - afb_arab
    - als_latn
    - amh_ethi
    - apc_arab_jord
    - apc_arab_leba
    - apc_arab_pale
    - apc_arab_syri
    - arb_arab
    - arq_arab
    - ars_arab
    - ary_arab
    - arz_arab
    - asm_beng
    - azj_latn
    - bam_latn
    - bcc_arab
    - bel_cyrl
    - ben_beng
    - ben_latn
    - bgc_deva
    - bho_deva
    - bos_latn
    - bra_deva
    - bsk_arab
    - btx_latn
    - bul_cyrl
    - cat_latn
    - ces_latn
    - ckb_arab
    - ckm_latn
    - cls_deva
    - cmn_hans
    - cmn_hant
    - dan_latn
    - deu_latn
    - dhd_deva
    - dje_latn
    - ekk_latn
    - ekp_latn
    - ell_grek
    - eng_latn
    - eus_latn
    - fao_latn
    - fin_latn
    - fra_latn_cana
    - fra_latn_fran
    - glg_latn
    - guj_gujr
    - hau_latn
    - haw_latn
    - heb_hebr
    - hin_deva
    - hin_latn
    - hrv_latn
    - hun_latn
    - hye_armn
    - ibo_latn
    - idu_latn
    - ind_latn
    - isl_latn
    - iso_latn
    - ita_latn
    - jav_latn
    - jpn_jpan
    - kan_knda
    - kan_latn
    - kat_geor
    - kaz_cyrl
    - kin_latn
    - kir_cyrl
    - kor_hang
    - lin_latn
    - lit_latn
    - luo_latn
    - mal_mlym
    - mar_deva
    - mkd_cyrl
    - mni_beng
    - mni_mtei
    - nag_latn
    - nld_latn
    - nno_latn
    - nob_latn
    - npi_deva
    - ory_orya
    - pan_guru
    - pcm_latn
    - pes_arab
    - plt_latn
    - pol_latn
    - por_latn_braz
    - por_latn_port
    - ron_latn
    - rus_cyrl
    - rwr_deva
    - sin_latn
    - sin_sinh
    - slk_latn
    - slk_latn_sari
    - slv_latn
    - slv_latn_cerk
    - slv_latn_prle
    - snd_arab
    - snd_deva
    - spa_latn_mexi
    - spa_latn_peru
    - spa_latn_spai
    - srp_cyrl
    - srp_cyrl_torl
    - srp_latn
    - srp_latn_torl
    - sun_latn
    - swe_latn
    - swh_latn
    - swv_deva
    - tam_latn
    - tam_taml
    - tel_latn
    - tel_telu
    - tgl_latn
    - tha_thai
    - tur_latn
    - uig_arab
    - ukr_cyrl
    - urd_arab
    - urd_latn
    - urh_latn
    - uzn_latn
    - vie_latn
    - yor_latn
    - yue_hant
    - zsm_latn
    - zul_latn

**Parallel (131)**

    - acm_arab
    - acq_arab
    - aeb_arab
    - afb_arab
    - als_latn
    - amh_ethi
    - apc_arab_jord
    - apc_arab_leba
    - apc_arab_pale
    - apc_arab_syri
    - arb_arab
    - arq_arab
    - ars_arab
    - ary_arab
    - arz_arab
    - asm_beng
    - azj_latn
    - bam_latn
    - bcc_arab
    - bel_cyrl
    - ben_beng
    - ben_latn
    - bgc_deva
    - bho_deva
    - bos_latn
    - bra_deva
    - bsk_arab
    - btx_latn
    - bul_cyrl
    - cat_latn
    - ces_latn
    - ckb_arab
    - ckm_latn
    - cls_deva
    - cmn_hans
    - cmn_hant
    - dan_latn
    - deu_latn
    - dhd_deva
    - dje_latn
    - ekk_latn
    - ekp_latn
    - ell_grek
    - eng_latn
    - eus_latn
    - fao_latn
    - fin_latn
    - fra_latn_cana
    - fra_latn_fran
    - glg_latn
    - guj_gujr
    - hau_latn
    - haw_latn
    - heb_hebr
    - hin_deva
    - hin_latn
    - hrv_latn
    - hun_latn
    - hye_armn
    - ibo_latn
    - idu_latn
    - ind_latn
    - isl_latn
    - iso_latn
    - ita_latn
    - jav_latn
    - jpn_jpan
    - kan_knda
    - kan_latn
    - kat_geor
    - kaz_cyrl
    - kin_latn
    - kir_cyrl
    - kor_hang
    - lin_latn
    - lit_latn
    - luo_latn
    - mal_mlym
    - mar_deva
    - mkd_cyrl
    - mni_beng
    - mni_mtei
    - nag_latn
    - nld_latn
    - nno_latn
    - nob_latn
    - npi_deva
    - ory_orya
    - pan_guru
    - pcm_latn
    - pes_arab
    - plt_latn
    - pol_latn
    - por_latn_braz
    - por_latn_port
    - ron_latn
    - rus_cyrl
    - rwr_deva
    - sin_latn
    - sin_sinh
    - slk_latn
    - slk_latn_sari
    - slv_latn
    - slv_latn_cerk
    - slv_latn_prle
    - snd_arab
    - snd_deva
    - spa_latn_mexi
    - spa_latn_peru
    - spa_latn_spai
    - srp_cyrl
    - srp_cyrl_torl
    - srp_latn
    - srp_latn_torl
    - sun_latn
    - swe_latn
    - swh_latn
    - swv_deva
    - tam_latn
    - tam_taml
    - tel_latn
    - tel_telu
    - tgl_latn
    - tha_thai
    - tur_latn
    - uig_arab
    - ukr_cyrl
    - urd_arab
    - urd_latn
    - urh_latn
    - uzn_latn
    - vie_latn
    - yor_latn
    - yue_hant
    - zsm_latn
    - zul_latn

</details>


### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [x] Is the "Main" variant of this task clearly denoted?
* [x] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [x] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog


### Citation

```text
@article{mrl-workshop-2025-global-piqa,
  title={Global {PIQA}: Evaluating Commonsense Reasoning Across 100+ Languages and Cultures},
  author={Tyler A. Chang and Catherine Arnett and Abdelrahman Sadallah and Abdelrahman Eldesokey and Abeer Kashar and Abolade Daud and Abosede Grace Olanihun and Adamu Labaran Mohammed and Adeyemi Praise and Adhikarimayum Meerajita Sharma and Aditi Gupta and Adril Putra Merin and Adwoa Bremang and Afitab Iyigun and Afonso Simplício and Ahmed Essouaied and Aicha Chorana and Akhil Eppa and Akintunde Oladipo and Akriti Kuri and Akshay Ramesh and Aleksei Dorkin and Alfred Malengo Kondoro and Alham Fikri Aji and Ali Eren Çetintaş and Allan Hanbury and Alou Dembele and Alp Niksarli and Álvaro Arroyo and Amin Bajand and Amol Khanna and Ana Chkhaidze and Ana Carolina Condez and Anamaria-Roberta Hartl and Andiswa Mkhonto and Andrew Hoblitzell and Andrew Tran and Angelos Poulis and Anirban Majumder and Anjali Chaudhary and Anna Vacalopoulou and Annette Kuuipolani Kanahele Wong and Annika Simonsen and Anton Kovalev and Anupam Nayak and Ashvanth S and Ayodeji Lana and Ayu Purwarianti and Bashar Alhafni and Benedict Busole and Bernard Ghanem and Bharti Nathani and Biljana Stojanovska Đurić and Blessing Ogundipe and Bolaotan Agbonile and Bragi Bergsson and Bruce Torres Fischer and Burak Tutar and Burcu Çınar and Cade Kane and Can Udomcharoenchaikit and Chadi Helwe and Chaithra Reddy Nerella and Chen Cecilia Liu and Chiamaka Nwokolo and Christopher Homan and Clément Sampebgo and Cristina España-Bonet and Cynthia Amol and Daeyoep Lee and Dan Saattrup Smart and Dana Arad and Daniil Dzenhaliou and Dasol Choi and David Liu and David Semedo and David Anugraha and Deborah Popoola and Deividas Mataciunas and Delphine Nyaboke and Dennis Owusu and Dhyuthy Krishna Kumar and Diogo Tavares and Diogo Glória-Silva and Divyanshu Goyal and DongGeon Lee and E. Kelly Buchanan and Ebele Nwamaka Anajemba and Egonu Ngozi Grace and Elena Mickel and Elias Herranen and Eliza Acharya and Eman Nisar and Emile Anand and Emmanuel Habumuremyi and Emuobonuvie Maria Ajiboye and Eryawan Presma Yulianrifat and Esther Adenuga and Ewa Rudnicka and Faith Itiola and Faran Taimoor Butt and Fareeha Fayyaz Sheikh and Fathima Thekkekara and Fatima Haouari and Faustin Nsengiyumva and Fenal Ashokbhai Ilasariya and Filbert Aurelian Tjiaranata and Firas Laakom and Francesca Grasso and Francesco Periti and Francesco Orabona and Gbenga Kayode Solomon and Genta Indra Winata and Gia Nghia Ngo and Gloria Udhedhe-oze and Gonçalo Vinagre and Gopi Naga Sai Ram Challagolla and Gorka Urbizu-Garmendia and Gouthami Vadithya and Guijin Son and Gulnaz Abdykadyrova and Gyan Swaroop Mohapatra and Hafeez Ullah and Hafsteinn Einarsson and Hai Hu and Hamidreza Saffari and Hamza Zaidi and Haopeng Zhang and Harethah Abu Shairah and Harry Vuong and Hele-Andra Kuulmets and Hitesh Laxmichand Patel and Houda Bouamor and Hwanjo Yu and Iben Nyholm Debess and İbrahim Ethem Deveci and Ikhlasul Akmal Hanif and Ikhyun Cho and Inês Vieira and Inês Calvo and Isaac Manzi and Ismael Illa Salifou and Ismail Daud and Ismail Yusuf and Itay Itzhak and Ivan Zhelyazkov and Ivan Belashkin and Ivan Spada and Jacob Brinton and Jafar Isbarov and Jaka Čibej and Jan Kocoń and Jan Cuhel and Jauza Krito and Jebish Purbey and Jennifer Za and Jennifer Mickel and Jenny Kunz and Jessica Ratovondranto and Jeyarajalingam Varsha and Jihae Jeong and Jimena Tena Dávalos and Jinu Lee and João Magalhães and John Seon Keun Yi and Jongin Kim and Joseph Chataignon and Joseph Marvin Imperial and Jubeerathan Thevakumar and Judith Land and Julia Alekseenko and Junchen Jiang and Jungwhan Kim and Kairit Sirts and Kamesh R and Kamesh V and Kanda Tshinu and Kätriin Kukk and Kaustubh Ponkshe and Kavsar Huseynova and Ke He and Kenneth Enevoldsen and Kent Joshua Alvarez and Kerem Zaman and Khalil Mrini and Kian Kyars and Komal Gour and Krishnakumar Lainitha and Krister Kruusmaa and Kunal Mukherjee and Kusum Chouhan and Laura Castro and Laura M. Porrino-Moscoso and Lenny Sivi Za Nzambi and Leshem Choshen and Levent Sencan and Lilja Øvrelid and Lisa Alazraki and Loretta Oma Jones and Lovina Ehimen-Ugbede and Luheerathan Thevakumar and Luxshan Thavarasa and Mahnoor Malik and Mamadou K. Keita and Mansi Jangid and Marco De Santis and Marcos Garcia and Marek Šuppa and Mariam D'Ciofalo and Marii Ojastu and Marium Attaullah and Maryam Sikander and Mausami Narayan and Maximos Skandalis and Mehak Mehak and Mehmet İlteriş Bozkurt and Melaku Bayu and Menan Velayuthan and Mhasilenuo Vizo and Michael Leventhal and Michał Marcińczuk and Mina Almasi and Mirna Potočnjak and Mithil Bangera and Mohammadamin Shafiei and Mohiba Ansari and Mridul Sharma and Mrityunjaya Indoria and Mughees Ur Rehman and Muhammad Ravi Shulthan Habibi and Murat Kolić and Murat Barkın Kınay and Nada Galant and Naina Singh Rathore and Naphat Permpredanun and Narada Maugin and Nathalie Norman and Nicholas Kluge Corrêa and Nikola Ljubešić and Nirmal Thomas and Nisansa de Silva and Nisheeth Joshi and Nitish Ponkshe and Nizar Habash and Nneoma Udeze and Noel Thomas and Noémi Ligeti-Nagy and Nouhoum Coulibaly and Odunayo Ogundepo and Odunayo Kareemat Buliaminu and Oghojafor Godswill Fejiro and Okechukwu God'spraise and Olanrewaju Samuel and Olaoye Deborah Oluwaseun and Olasoji Akindejoye and Olga Snissarenko and Onyinye Anulika Chiemezie and Orkun Kınay and Osman Tursun and Oyelade Oluwafemi Joshua and Oyesanmi Fiyinfoluwa and Pablo Rodríguez and Pablo Gamallo and Palak Arora and Pedro Valente and Peter Rupnik and Philip Oghenesuowho Ekiugbo and Prakhar Agarwal and Pramit Sahoo and Prokopis Prokopidis and Pua Niau-Puhipau and Quadri Yahya and Rachele Mignone and Raghav Singhal and Rahul Raja and Ram Mohan Rao Kadiyala and Raphael Merx and Rasmus Larsen and Ratnavel Rajalakshmi and Rishav Ghosh and Romina Oji and Ron Kekeha Solis and Rui Guerra and Rushikesh Zawar and Sa'ad Nasir Bashir and Saeed Alzaabi and Sahil Sandeep and Sai Pavan Batchu and Sai Sandeep Kantareddy and Saleha Muzammil and Salsabila Zahirah Pranida and Sam Buchanan and Samuel Rutunda and Sander Land and Sarah Sulollari and Sardar Ali and Saroj Sapkota and Sarveswaran Kengatharaiyer and Saulius Tautvaisas and Sayambhu Sen and Sayantani Banerjee and Sebastien Diarra and Segun Afolayan and Senthilnathan M and Sewoong Lee and Shaan Shah and Shankar Venkitachalam and Sharifa Djurabaeva and Sharon Ibejih and Shivanya Shomir Dutta and Siddhant Gupta and Silvia Paniagua Suárez and Sina Ahmadi and Sivasuthan Sukumar and Siyuan Song and Snegha A and Sokratis Sofianopoulos and Sona Elza Simon and Sonja Benčina and Sophie Gvasalia and Sphurti More and Spyros Dragazis and Stefan Milosavljević and Stephan P. Kaufhold and Suba S and Sultan Alrashed and Surangika Ranathunga and Taiga Someya and Taja Kuzman Pungeršek and Tal Haklay and Tasi'u Jibril and Tatsuya Aoyama and Tea Abashidze and Terenz Jomar Dela Cruz and Terra Blevins and Themistoklis Nikas and Theresa Idoko and Thu Mai Do and Tilek Chubakov and Tina Munda and Tobiloba Owoeye and Tommaso Gargiani and Uma Rathore and Uni Johannesen and Uwuma Ugwu and Vallerie Alexandra Putra and Vanya Bannihatti Kumar and Varvara Arzt and Vasily Konovalov and Vasudevan Nedumpozhimana and Viktoria Ondrejova and Viktoryia Horbik and Vishnu Vardhan Reddy Kummitha and Vuk Dinić and Walelign Sewunetie and Winston Wu and Xiaojing Zhao and Yacouba Diarra and Yaniv Nikankin and Yash Mathur and Yash Bagla and Yeshil Bangera and Yixi Chen and Yiyuan Li and Yolanda Xavier and Yonatan Belinkov and Zaid Alyafeai and Zhargal Batozargalova and Zhengyang Shan and Zhi Rui Tam and Zilu Tang and Zuzana Nadova and Baber Abbasi and Stella Biderman and David Stap and Duygu Ataman and Fabian Schmidt and Hila Gonen and Jiayi Wang and David Ifeoluwa Adelani},
  journal={Preprint},
  year={2026},
  url={https://arxiv.org/abs/2510.24081},
}
```

