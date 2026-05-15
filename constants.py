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


test_models = {'meta-llama-3':['meta-llama-3-70b', 'meta-llama-3-70b-instruct'],
               'qwen2':['qwen2-72b', 'qwen2-72b-instruct'],
               'yi-1.5':['yi-1.5-34b', 'yi-1.5-34b-chat'],
               'olmo':['olmo-7b'],
               'smollm':['smollm-1.7b', 'smollm-1.7b-instruct'],
               'gemma2': ['gemma-2-9b', 'gemma-2-9b-it']}

delete_models = {'meta-llama-3':['calme-2.2-llama3-70b',
                                    'calme-2.3-llama3-70b',
                                    'calme-2.4-llama3-70b',
                                    'dolphin-2.9.1-llama-3-70b',
                                    'higgs-llama-3-70b',
                                    'llama-3-70b-instruct-v0.1',
                                    'llama-3-sauerkrautlm-70b-instruct',
                                    'llama3-openbiollm-70b',
                                    'meta-llama-3-70b',
                                    'meta-llama-3-70b-instruct'],
                'qwen2':['calme-2.1-qwen2-72b',
                            'calme-2.2-qwen2-72b',
                            'calme-2.3-qwen2-72b',
                            'dolphin-2.9.2-qwen2-72b',
                            'magnum-72b-v1',
                            'magnum-v1-72b',
                            'magnum-v2-72b',
                            'orca_mini_v7_72b',
                            'qwen2-72b',
                            'qwen2-72b-instruct'],
                'yi-1.5':['blossom-v5.1-34b',
                            'dolphin-2.9.1-yi-1.5-34b',
                            'yi-1.5-34b',
                            'yi-1.5-34b-chat'],
                'olmo':['olmo-7b'],
                'smollm':['smollm-1.7b', 'smollm-1.7b-instruct',
                        'smollm-1.7b-instruct-ifeval'],
                'gemma2':['gemma-2-9b-it-dpo',
                            'gemma-2-9b-it-simpo',
                            'gemma-2-9b-it-wpo-hb',
                            'gemma-2-9b-moth',
                            'gemma2-9b-it-psy10k-mental_health',
                            'gemma2-9b-it-simpo-infinity-preference',
                            'gemma2-9b-it-train6',
                            'lambda-gemma-2-9b-dpo',
                            'n3n_gemma-2-9b-it_20241029_1532',
                            'n3n_gemma-2-9b-it_20241110_2026',
                            'magnum-v3-9b-customgemma2']}

Y_names_tidy = {'gsm8k':'GSM8k','ifeval':'IFEval','hellaswag':'HellaSwag','mmlu':'MMLU','arc':'ARC',
                'truthfulqa':'TruthfulQA','winogrande':'Winogrande','bbh':'BBH','math':'MATH',
                'mmlu-pro':'MMLU-Pro','gpqa':'GPQA','musr':'MuSR'}

Y_names = {}
Y_names[0] = ['math','ifeval','hellaswag','bbh','mmlu-pro','mmlu','arc','truthfulqa','gsm8k','winogrande','gpqa','musr']
Y_names[1] = ['gsm8k','truthfulqa','hellaswag','mmlu','arc','winogrande']
Y_names[2] = ['math','ifeval','bbh','mmlu-pro','gpqa','musr']

B = 10000
lrs = [.1,.05,.01]
scheduler_factors = [.999,.99]
reps = 5
n_epochs = 20000
random_seed = 42