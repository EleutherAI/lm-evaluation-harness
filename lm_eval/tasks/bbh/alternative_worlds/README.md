| Task               | Prompt Variation  | Output Variation  | Option in Sample |
| :-----------------:| :---------------: | :---------------: |:---------------: |
| boolean_expression | Yes               | Yes               | No               |
| causal_judgement   | Yes               | Yes               | Yes              |
| date_understanding | Yes               | Yes               | Yes              |
| disambiguation_qa  | Yes               | Yes               | Yes              |
| dyck_languages     | Yes               | No                | No               |
| formal_fallacies   | Yes               | Yes               | Yes              |
| geometric_shapes   | Yes               | Yes               | Yes              |
| hyperbaton         | Yes               | Yes               | Yes              |
| logical_deduction_five_objects| Yes    | Yes               | Yes              |
| logical_deduction_seven_objects| Yes   | Yes               | Yes              |
| logical_deduction_three_objects| Yes   | Yes               | Yes              |
| movie_recommendation| Yes              | Yes               | Yes              |
| multistep_arithmetic_two| Yes          | No                | No               |
| navigate           | Yes               | Yes               | Yes              |
| object_counting    | Yes               | No                | No               |
| penguins_in_a_table| Yes               | Yes               | Yes              |
| reasoning_about_colored_objects| Yes   | Yes               | Yes              |
| ruin_names         | Yes               | Yes               | Yes              |
| salient_translation_error_detection| Yes| Yes              | Yes              |
| snarks             | Yes               | Yes               | Yes              |
| sports_understanding| Yes              | Yes               | No               |
| temporal_sequences | Yes               | Yes               | Yes              |
| tracking_shuffled_objects_five_objects| Yes| Yes           | Yes              |
| tracking_shuffled_objects_seven_objects| Yes| Yes          | Yes              |
| tracking_shuffled_objects_three_objects| Yes| Yes          | Yes              |
| web_of_lies        | Yes               | Yes               | No               |
| word_sorting       | Yes               | No                | No               |


Notes:
- `web_of_lies` already starts with `Question: `
- Tasks with options are `Options: (A) ...` (multiple choice) or `Options: - ...` (binary choice)