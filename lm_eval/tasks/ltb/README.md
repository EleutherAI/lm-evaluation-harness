# Last Translation Benchmark v1

`ltb_v1` evaluates translation generation on the official **LTBv1-eval** subset:
911 text-only examples with 1,693 human-written verification rules. It preserves
the upstream translation prompt and verifies each rule separately. Rules and
baseline translations are never included in the translation prompt.

- [Main benchmark page](https://last-translation-benchmark.vilda.net/)
- [Benchmark repository](https://github.com/zouharvi/last-translation-benchmark)
- [Dataset](https://huggingface.co/datasets/zouhar/last-translation-benchmark)
- [Paper](https://arxiv.org/abs/2609.04173)

The YAML config uses `dataset_path: zouhar/last-translation-benchmark` and loads
through Hugging Face Datasets. The release is pinned to dataset revision
`a483825ddbe2d7756f5bdfb1e4f611bee9026c4c`, file `data/v1.json`.
The Hub exposes the release as the `train` split; this task uses that split for
testing, selecting the `LTBv1-eval` tag. The full LTBv1 release also contains media
and examples outside the evaluation subset. Evaluation is zero-shot.

## Running

Install the model backend and the optional judge client:

```bash
pip install -e '.[hf,ltb]'
export LTB_JUDGE_API_KEY='your-api-key'
lm_eval --model hf --model_args pretrained=YOUR_MODEL \
  --tasks ltb_v1 --apply_chat_template --batch_size auto \
  --log_samples --output_path results/ltb_v1
```

The default judge is `google/gemini-3.1-pro-preview`, matching the upstream
leaderboard's judge model, served through OpenRouter. The judge receives the
source, generated translation, and one rule per request. Judge calls are made
only during scoring. Environment variables configure an alternative
OpenAI-compatible chat-completions service:

| Variable | Default |
| --- | --- |
| `LTB_JUDGE_API_KEY` | Required for scoring |
| `LTB_JUDGE_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LTB_JUDGE_MODEL` | `google/gemini-3.1-pro-preview` |
| `LTB_JUDGE_WORKERS` | `8` concurrent examples |
| `LTB_OUTPUT_DIR` | New `ltb-results-*` directory in the current directory |
| `LTB_RESPONSE_FORMAT` | `text`; use `harmony` for raw gpt-oss output |

For gpt-oss with vLLM, `LTB_RESPONSE_FORMAT=harmony` extracts the final channel
for judging and submission export. Leave the backend's `think_end_token` unset
and preserve special tokens so this task can recognize the channel boundaries.
Leave the backend's `enable_thinking` unset as well: explicitly enabling its
generic thinking mode requires a `think_end_token`, which would strip the raw
channel headers before this task sees them. The native gpt-oss chat template
controls its own reasoning behavior.
Raw responses, including analysis, remain in the harness samples. Responses
without a final channel become `null` translations and fail instead of sending
unfinished reasoning to the judge. This extraction runs in `process_results`,
so it does not apply to `--predict_only` output.

Scoring saves `ltb_submission.json` before making judge requests and writes
`judge_responses.jsonl` after every completed request. The JSONL records contain
the dataset ID, rule index, exact request, full API response (including reasoning
and token usage when returned), parsed verdict, and per-example score. API errors
record the exception type and HTTP status; truncated responses are saved before
scoring aborts. Each record is flushed to disk so completed calls survive a later
timeout. Credentials and request headers are not logged. Existing judge logs are
never overwritten; use a fresh output directory for each evaluation.

Set `LTB_OUTPUT_DIR` explicitly to place the submission and judge log in your
chosen results directory. This logging runs during scoring; `--predict_only`
continues to use the harness sample files.

Use `--limit 5` for a small trial. A full evaluation makes up to 1,693 judge
requests, which may incur API charges. Generation allows 4,096 tokens without
newline stop sequences, preserving multi-paragraph translations. Adjust the
token limit through `--gen_kwargs max_gen_toks=8192` if needed.

To generate and save translations without calling a judge, add `--predict_only`
and provide `--output_path`. Once the Hugging Face dataset is cached, set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to load it without network access.

## Metric

`ltb_pass_rate` is the fraction of examples for which **all** verification rules
pass, on a 0–1 scale. Each example has equal weight regardless of rule count.
Empty translations fail. Verdict parsing follows the official leaderboard's
permissive pass/fail parsing, including treating unrecognized responses as fail.
API exceptions after retries and truncated judge responses abort scoring so that
service failures do not silently alter the score or denominator.
On failure, scheduling stops, pending examples are cancelled, and workers skip
remaining rules. Requests already in flight finish (including SDK retries) and
are logged before the judge client and logs close; aborting is not instantaneous.

Changing the judge changes the evaluation protocol; report the judge model and
endpoint with results. These locally computed scores are not leaderboard
submissions. The official scorer uses its own API/cache infrastructure, so even
the default model does not guarantee identical verdicts across runs.

## Attribution

Dataset: CC BY 4.0. Upstream prompt and parser code: MIT, copyright (c) 2026
Vilém Zouhar. See [LICENSE](LICENSE) for the upstream code notice.

```bibtex
@misc{zouhar2026translationbenchmark,
      title={Last Translation Benchmark},
      author={Vilém Zouhar and Niyati Bafna and Mukund Choudhary and Maike Züfle and Sara Rajaee and Pinzhen Chen and Jannis Vamvas and Sara Papi and Ona de Gibert and Bhavitvya Malik and Eliya Habba and Orfeas Menis Mastromichalakis and Patrícia Schmidtová and Michelle Wastl and Sheriff Issaka and Leshem Choshen and Stella Biderman and Antonis Anastasopoulos and Jan Niehues and Rico Sennrich and Mrinmaya Sachan and Ondřej Bojar and Kenton Murray and Jörg Tiedemann and Alham Fikri Aji and Philipp Koehn and Christof Monz and Alexandra Birch and Sowmya Vajjala and Chalamalasetti Kranti and Cristina España-Bonet and Nobin Sarwar and David Kaczér and Shunta Asano and Malik Marmonier and Daban Q. Jaff and Vaisakhi Mishra and Hend Al- Khalifa and Gabriele Sarti and Sourajit Saha and Nils Rehlinger and Juan Daniel Cuervo Villa and Jonathan Tonglet and Saugata Purkayastha and Dominik Macháček and Jagannathan Ramanujam and Heejin Do and Zuzana Nadova and Fred Philippy and Fabian Retkowski and Maria Lymperaiou and Silvia Casola and Hanna Yukhymenko and Shubhashis Roy Dipta and Sangwon Ryu and Andrés Jerez and Ron Keinan and Shuaib Shuaib Yusuf and Avantica Vempati and Maria Carmen Staiano and Sukannya Purkayastha and Adrian Cosma and Vitalii Babenko and Erivan Inan and Aviral Nigam and Wafa Aissa and Fatima Haouari and Venkata Prasanth Kumar Gummadi and Mehdi Jafarzadeh and Valentin Scourneau and Lukas Edman and Kaiser Sun and Shaomu Tan and Mohammad Sadegh Gholizadeh and Johannes-Rudolf David and Dipankar Srirag and Javier García Gilabert and Ruta Binkyte and Manar Ali and Ana-Maria Bucur and Sabry E. Farrag and Youssef Saber and Yihong Liu and Jean Maillard and Cojocaru Nicoleta and Xiaochuang Yuan and Sina Ahmadi and Philipp Mondorf and Kaustubh Dhole and Roman Wixinger and Shenbin Qian and Manuel Tuor and Sergey Troshin and Jonathan Yahav and Fida Mohammad Thoker and Amir Arsalan Rezapour and Lance Calvin Lim Gamboa and Manon Reusens and Kätriin Kukk and Koel Dutta Chowdhury and Giuseppe Gallipoli and Christian Hoang and Shaswati Saha and Seth Aycock and Jan Kocoń and Bo Chen and Linh Vu and Vatsal Venkatkrishna and Arafat Ahsan and Luan Thanh Nguyen and Hassan Soliman and Daryna Dementieva and Theresia Veronika Rampisela and Ngoc Quynh Tram Do and Marius Huber and Kazuki Egashira and Azmine Toushik Wasi and Vladislav Poritski and Mike Zhang and Deep Shah and Paul Gavrikov and Luis Frentzen Salim and David Africa and R. Damanhuri and Bello Umar Bello and Anumit Garg and Gengyu Rao and Pawan Sasanka Ammanamanchi and Kamile Dementaviciute and Andrianos Michail and L D M S Sai Teja and Dawei Zhu and Yi Fan and Wei Liu and Farhan Farsi and Elias Herranen and Sankalan Pal Chowdhury and Karen Sanchez and Farzad Shami and Ashok Urlana and Zimu Wang and Tomasz Limisiewicz and Priyaranjan Pattnayak and Marii Ojastu and Hongbin Na and Emilian Radoi and Chenyi Zhao and Carlos Hinojosa and Andrea Gregor de Varda and Zaid Alyafeai and Reem Alzahrani and Nehal Kathrotia and Alex Flückiger and Ulysses Sekai Tully Carr and Jimson Paulo Layacan and Guy Kaplan and Ritwik Tiwari and Rishit Dagli and Oksana Volchek and Isaac R Caswell and Bowen Yi and Blanka Kövér and Amir Hossein Yari and Aicha Chorana and Zhengxiang Wang and Selja Keränen and Samuel Simko and Joy Olusanya and Jenny Chim and Enzo Doyen and Vivek Harsha Lakkamaneni and Sophia Conrad and Pouya Sadeghi and Panayiotis Panayiotou and Luis Lara and Jannatul Nayem and Eran Yahav and Debanshu Das and Antonia Karamolegkou and Anmol Goel and Aishik Mandal and Tommaso Cerruti and Raoyuan Zhao and Mykola Haltiuk and Thura Aung and Naser Almousa and Amir Hossein Kargaran and Rachel Bawden and Qiaoyuan Zheng and Mateusz Lango and Beni Egressy and Fidel Rodríguez Velásquez and Natchapon Jongwiriyanurak and Minh Ngoc Do and Marco Gaido and Lena Libon and Dzmitry Kuzmin and Badal Nyalang and Antoine Taroni and Andrei Niculae and Abdulaziz Nura Kani and Rushikesh Zawar and Marek Šuppa and Beatrice Savoldi and Andreas Simons and Rayyan Merchant and Ilai Yaron Levy and Francesco Pinto and Ziyi Yang and Yolanda Xavier and Samuel Frontull and Muhammad Ravi Shulthan Habibi and Kenneth Enevoldsen and Harris Abdul Majid and Francesca Padovani and Tim Graf and Tatiana Bielakova and Sharifa Djurabaeva and Shaoxiong Ji and Raia Abu Ahmad and Pavel Stepachev and Jirui Qi and Ayush Sunil Munot and Alireza Pakniat and Ayla Rigouts Terryn and Yuxing Lu and Yurii Paniv and Xiyan Fu and Tosin Adewumi and Sunisth Kumar and Stéphane J. P. S. Thunus and Shree Harsha Bokkahalli Satish and Shayan Bali and Prakhar Gupta and Papa Abdou Karim Karou Diallo and Matija Akrap and Marko Culjak and Kristýna Onderková and Joseph Attieh and Esrael Teferi Tensay and Elisabeth Fittschen and Benoît Sagot and Jingwei Ni and Yu Fan},
      year={2026},
      eprint={2609.04173},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.04173},
}
```
