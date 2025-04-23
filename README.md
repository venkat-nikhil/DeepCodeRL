# DeepCodeRL
Code Generation via Iterative Reinforcement Learning

//For GPUs with small vram <6GB

python main.py --run_sft --run_initial_rl --batch_size 1 --gradient_accumulation_steps 32 --max_length 512 --use_fp16 --use_8bit --gradient_checkpointing --use_lora

//For GPUs with large vram >24gb

python main.py --run_sft --run_initial_rl --batch_size 4 --gradient_accumulation_steps 8 --max_length 1024 --use_fp16 --use_lora

// run only SFT stage

[python main.py --run_sft --batch_size 1 --gradient_accumulation_steps 16 --max_length 1024 --use_fp16 --gradient_checkpointing --use_lora]

// run only initial_rl stage

python main.py --run_initial_rl --sft_model_path output/sft/final_model/  --batch_size 1 --gradient_accumulation_steps 32 --max_length 512 --use_fp16 --use_8bit --gradient_checkpointing --use_lora


python main.py --run_initial_rl --sft_model_path output/sft/final_model/  --batch_size 8 --gradient_accumulation_steps 32 --max_length 1024  --num_epochs 1

python main.py --run_initial_rl --sft_model_path output/sft/final_model/  --batch_size 8 --gradient_accumulation_steps 32 --rl_max_length 8192 --num_epochs 1 --use_fp16