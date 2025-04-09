# DeepCodeRL
Code Generation via Iterative Reinforcement Learning


python SFT.py --json_path data/processed_codeforces/filtered_solutions_py_decontaminated_Python3Code_all_solutions.json --processed_data_dir data/processed_codeforces/ --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --batch_size 1 --gradient_accumulation_steps 16 --max_length 1024 --use_lora --use_fp16 --gradient_checkpointing