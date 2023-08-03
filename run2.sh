# python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
#                 --task kolegal_criminalcase  --num_fewshot 0
# python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
#                 --task kolegal_criminalcase  --num_fewshot 5
# python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
#                 --task kolegal_criminalcase  --num_fewshot 10

python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
	--task ko_en_translation --num_fewshot 5

# test : numbers 
#python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
#                --task kolegal_legalcase  --num_fewshot 0 
# python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
#                 --task kolegal_legalcase  --num_fewshot 5
# python3 -W ignore main.py --model gpt2 --model_args pretrained=EleutherAI/polyglot-ko-1.3B \
#                 --task kolegal_legalcase  --num_fewshot 10
