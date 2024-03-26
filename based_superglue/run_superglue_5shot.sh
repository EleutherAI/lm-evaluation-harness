
CUDA_VISIBLE_DEVICES=0 python launch.py --task boolq --task cb --task copa --task multirc --task record --task rte --task wic --task wsc --num_fewshot 5 --model "hazyresearch/based-360m" &

CUDA_VISIBLE_DEVICES=1 python launch.py --task boolq --task cb --task copa --task multirc --task record --task rte --task wic --task wsc --num_fewshot 5 --model "hazyresearch/mamba-360m" &

CUDA_VISIBLE_DEVICES=2 python launch.py --task boolq --task cb --task copa --task multirc --task record --task rte --task wic --task wsc --num_fewshot 5  --model "hazyresearch/attn-360m" &

CUDA_VISIBLE_DEVICES=6 python launch.py --task boolq --task cb --task copa --task multirc --task record --task rte --task wic --task wsc --num_fewshot 5 --model "hazyresearch/based-1b" &

CUDA_VISIBLE_DEVICES=5 python launch.py --task boolq --task cb --task copa --task multirc --task record --task rte --task wic --task wsc --num_fewshot 5 --model "hazyresearch/mamba-1b" &

CUDA_VISIBLE_DEVICES=7 python launch.py --task boolq --task cb --task copa --task multirc --task record --task rte --task wic --task wsc --num_fewshot 5 --model "hazyresearch/attn-1b" &
