export PYTHONPATH=$(pwd)/../SmartChat/script
export LOGURU_LEVEL=CRITICAL


ls -d ./data/B1_gpt4.1mini/* | xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio {} gpt4.1mini > B1_gpt4.1mini.coverage
ls -d ./data/B1_Llama3.3-70B/B1_Llama3.3-70B_* | xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio {} Llama3.3-70B > B1_Llama3.3-70B.coverage
ls -d ./data/B1_gemini-1.5-flash/B1_gemini-1.5-flash_* | xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio {} gemini-1.5-flash > B1_gemini-1.5-flash.coverage
ls -d ./data/B1_mixtral-8x7b_8192/*| xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio  {} mixtral-8x7b > B1_mixtral-8x7b_8192.coverage
ls -d ./data/B1_gpt4omini/* | xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio {} gpt4omini > B1_gpt4omini.coverage
ls -d ./data/B1_Llama3-8B/B1_Llama3-8B_* | xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio {} Llama3-8B  > B1_Llama3-8B.coverage
ls -d ./data/B1_Llama3-70B/B1_Llama3-70B_* | xargs -I {} poetry --directory ../SmartChat/script run genai4fuzz seed_coverage_ratio {} Llama3-70B > B1_Llama3-70B.coverage
