import numpy as np

### Lower bounds for benchmarks (computed using 1/#choices)
# for Open LLM LB 2, check https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about
BBH_ns ={'bbh_boolean_expressions':250,'bbh_causal_judgement':187,'bbh_date_understanding':250,'bbh_disambiguation_qa':250,
         'bbh_formal_fallacies':250,'bbh_geometric_shapes':250,'bbh_hyperbaton':250,'bbh_logical_deduction_five_objects':250,
         'bbh_logical_deduction_seven_objects':250,'bbh_logical_deduction_three_objects':250,'bbh_movie_recommendation':250,
         'bbh_navigate':250,'bbh_object_counting':250,'bbh_penguins_in_a_table':146,'bbh_reasoning_about_colored_objects':250,
         'bbh_ruin_names':250,'bbh_salient_translation_error_detection':250,'bbh_snarks':178,'bbh_sports_understanding':250,
         'bbh_temporal_sequences':250,'bbh_tracking_shuffled_objects_five_objects':250,'bbh_tracking_shuffled_objects_seven_objects':250,
         'bbh_tracking_shuffled_objects_three_objects':250,'bbh_web_of_lies':250}
BBH_cs ={'bbh_boolean_expressions':1/2,'bbh_causal_judgement':1/2,'bbh_date_understanding':1/6,'bbh_disambiguation_qa':1/3,
         'bbh_formal_fallacies':1/2,'bbh_geometric_shapes':1/11,'bbh_hyperbaton':1/2,'bbh_logical_deduction_five_objects':1/5,
         'bbh_logical_deduction_seven_objects':1/7,'bbh_logical_deduction_three_objects':1/3,'bbh_movie_recommendation':1/6,
         'bbh_navigate':1/2,'bbh_object_counting':1/19,'bbh_penguins_in_a_table':1/5,'bbh_reasoning_about_colored_objects':1/18,
         'bbh_ruin_names':1/6,'bbh_salient_translation_error_detection':1/6,'bbh_snarks':1/2,'bbh_sports_understanding':1/2,
         'bbh_temporal_sequences':1/4,'bbh_tracking_shuffled_objects_five_objects':1/5,'bbh_tracking_shuffled_objects_seven_objects':1/7,
         'bbh_tracking_shuffled_objects_three_objects':1/3,'bbh_web_of_lies':1/2}

MUSR_ns ={'musr_murder_mysteries':250,
          'musr_object_placements':256,
          'musr_team_allocation':250}
MUSR_cs ={'musr_murder_mysteries':1/2,
          'musr_object_placements':1/5,
          'musr_team_allocation':1/3}

lower_bounds = {'mmlu':.25,
                 'hellaswag':.25,
                 'winogrande':.5,
                 'gsm8k':0,
                 'arc':.25,
                 'truthfulqa':.31, #this number is computed by loading the leaderboard data and computing the 1st percentile of scores
                 'ifeval':0,
                 'math':0,
                 'mmlu-pro':.1,
                 'bbh':np.sum([BBH_ns[k]*BBH_cs[k] for k in BBH_cs.keys()])/np.sum([BBH_ns[k] for k in BBH_cs.keys()]),
                 'gpqa':.25,
                 'musr':np.sum([MUSR_ns[k]*MUSR_cs[k] for k in MUSR_cs.keys()])/np.sum([MUSR_ns[k] for k in MUSR_cs.keys()])}