export PYTHONPATH=$(pwd)/..
unset LOGURU_LEVEL

ls -d ../B1_gpt4.1mini/B1_gpt4.1mini_* | xargs -I {} poetry run genai4fuzz seed_metrics {} gpt4.1mini > B1_gpt4.1mini.metrics  2> output_b1_gpt4.1mini.log
ls -d ../B1_Llama3.3-70B/B1_Llama3.3-70B_* | xargs -I {} poetry run genai4fuzz seed_metrics {} Llama3.3-70B > B1_Llama3.3-70B.metrics  2> output_b1_Llama3.3-70B.log
ls -d ../B1_mixtral-8x7b_8192/B1_mixtral-8x7b_* | xargs -I {} poetry run genai4fuzz seed_metrics {} mixtral-8x7b > B1_mixtral-8x7b_8192.metrics  2> output_b1_mixtral-8x7b.log
ls -d ../B1_Llama3-70B/B1_Llama3-70B_* | xargs -I {} poetry run genai4fuzz seed_metrics {} Llama3-70B > B1_Llama3-70B.metrics  2> output_b1_Llama3-70B.log
ls -d ../B1_gpt4omini/B1_gpt4omini_* | xargs -I {} poetry run genai4fuzz seed_metrics {} gpt4omini > B1_gpt4omini.metrics  2> output_b1_gpt4omini.log
ls -d ../B1_Llama3-8B/B1_Llama3-8B_* | xargs -I {} poetry run genai4fuzz seed_metrics {} Llama3-8B > B1_Llama3-8B.metrics  2> output_b1_Llama3-8B.log
ls -d ../B1_gemini-1.5-flash/B1_gemini-1.5-flash_* | xargs -I {} poetry run genai4fuzz seed_metrics {} gemini-1.5-flash > B1_gemini-1.5-flash.metrics  2> output_b1_gemini-1.5-flash.log