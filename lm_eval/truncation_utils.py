from typing import Dict, Union

from lm_eval.utils import simple_parse_args_string


def process_truncation_args(args: Dict[str, str]) -> Dict[str, Union[str, int, bool]]:
    default_args = {
        "how": "no",
        "on": "tokens",
        "side": "left",
        "keep_first": False,
        "max_symbols": 2048,
        "max_new_symbols": 256,
    }
    if args:
        args = simple_parse_args_string(args)
        default_args.update(args)
    return default_args


def unpack_group(lst):
    return [elem for pack in lst for elem in pack]


def group_dicts(req, skip_system):
    groups = []
    if skip_system:
        groups.extend([[req[0]]])  # system instaruction goes as separate list
    # split remaining seq into pairs user-item
    for first in range(skip_system, len(req) - 1, 2):
        groups.extend([[req[first], req[first + 1]]])
    groups.extend([[req[-1]]])
    return groups


def tokenize_sequence(seq, tokenizer, add_special_tokens, symbols):
    # TODO: make sure it works for API and vLLM tokenizers
    if symbols:
        # consider 1 symbol = 1 token, for models with no tokenizer the only option
        return seq
    return tokenizer(seq, add_special_tokens=add_special_tokens)["input_ids"]


def apply_chat_template(seq, tokenizer, chat_template, add_generation_prompt, tokenize):
    # TODO: definitely work only for HF models
    if tokenizer is None:
        return chat_template(seq)
    return tokenizer.apply_chat_template(
        seq, add_generation_prompt=add_generation_prompt, tokenize=tokenize
    )


def instance_type(func):
    def wrapper(request, **kwargs):
        if len(request) == 1:
            return func(
                main=request[0], instance_type="loglikelihood_rolling", **kwargs
            )
        elif request[0] == "":
            return func(main=request[1], instance_type="acc_mutual_info", **kwargs)
        elif isinstance(request[1], dict):
            return func(main=request[0], instance_type="generate_until", **kwargs)
        elif isinstance(request[0], list) and isinstance(request[1], (str, int, float)):
            return func(
                main=request[0],
                additional=request[1],
                instance_type="loglikelihood",
                **kwargs,
            )

    return wrapper


def context_type(func):
    def wrapper(main, additional="", instance_type="generate_until", **kwargs):
        if isinstance(main, str):
            return func(main, additional, instance_type, "string", **kwargs)
        elif isinstance(main, list):
            if isinstance(main[0], str):
                return func(main, additional, instance_type, "no_template", **kwargs)
            if isinstance(main[0], dict):
                if isinstance(main[-1]["content"], list):
                    return func(
                        main, additional, instance_type, "chat_template", **kwargs
                    )
                else:
                    return func(main, additional, instance_type, "multiturn", **kwargs)

    return wrapper


def fewshots_truncation(
    request,
    target,
    instance_type,
    context_type,
    first_system,
    tokenizer,
    truncation_args,
    chat_template,
    add_special_tokens,
    max_new_tokens,
    max_length,
):
    if context_type == "string":
        return request, "nothing"
    if not len(request):
        return request, "empty"
    # do not cut off system prompt, so skip it if any
    skip_system = int(first_system)
    # whether to preserve the first element for truncation
    skip_first = int(truncation_args["keep_first"])
    # small hack, with no tokenizer this case may be reduced to no chat template one

    if len(request) and isinstance(request[0], list):
        req = []
        for lst in request:
            if len(lst) == 1:
                req.extend([lst[0]["content"]])
            else:
                req.extend([lst[0]["content"] + lst[1]["content"]])
    else:
        # do not change the initial sequence
        req = request[:]

    if context_type == "no_template":
        # append target for loglikelihood/multiple-choice, otherwise add empty string
        req[-1] += target
        # minimal length of request to truncate anything, +1 is for doc itself
        min_number_elements = skip_system + skip_first + 1
        # if skip_first and zero-shot = error
        if len(req) < min_number_elements:
            return request, "bad params"
        # TODO: do not tokenize the entire seq, tokenize shot by shot
        tokens = tokenize_sequence(
            req, tokenizer, add_special_tokens, truncation_args["on"] == "symbols"
        )
        # remaining for user prompt tokens, for generation tasks subtract tokens to be generated
        # TODO: take into account model type (seq2seq do not need subtraction)
        remain_tokens = max_length - max_new_tokens * int(
            instance_type == "generate_until"
        )
        # accumulate the total sum
        sum_seq = 0
        if truncation_args["side"] == "right":
            # skip system prompt
            start = skip_system
            # keep doc (question from test set) and may keep last shot
            end = len(tokens) - skip_first - 1
            reverse = 1
        else:
            # consider left to be default option
            # may skip first shot and skip system prompt
            start = skip_system + skip_first
            # keep doc (question from test set)
            end = len(tokens) - 1
            reverse = -1
        # minimal amount of tokens decided by the user
        sum_seq += sum(map(len, tokens[:start])) + sum(map(len, tokens[end:]))
        if sum_seq > max_length:
            return request, "error"
        result = 0
        for seq in tokens[start:end][::reverse]:
            sum_seq += len(seq)
            if sum_seq > remain_tokens:
                break
            result += 1
        # final = system_prompt + shots + keep_first + doc (right)
        if truncation_args["side"] == "right":
            final = (
                request[:skip_system]
                + request[skip_system : skip_system + result]
                + request[-2 : -2 + skip_first]
                + request[-1:]
            )
        # final = system_prompt + keep_first + shots + doc (left)
        else:
            final = (
                request[:skip_system]
                + request[skip_system : skip_system + skip_first]
                + request[-1 - result : -1]
                + request[-1:]
            )

        return final, result + skip_first
    elif context_type == "chat_template":
        if truncation_args["on"] != "symbols":
            if first_system:
                system = [req[0]]
                system_tokens = apply_chat_template(
                    system, tokenizer, chat_template, False, True
                )
                total_tokens = apply_chat_template(
                    [req[0], {"role": "user", "content": "".join(req[-1]["content"])}],
                    tokenizer,
                    chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            else:
                system_tokens = []
                total_tokens = apply_chat_template(
                    [{"role": "user", "content": "".join(req[-1]["content"])}],
                    tokenizer,
                    chat_template,
                    add_generation_prompt=True,
                    tokenize=True,
                )
            user_tokens = tokenize_sequence(
                ["".join(req[-1]["content"])], tokenizer, add_special_tokens, False
            )[0]
            # offset = system prompt tokens + all special and generation prompt tokens that will always be in input
            offset = len(system_tokens) + (
                len(total_tokens) - len(system_tokens) - len(user_tokens)
            )
            # with no chat template there could be only two (system, user) or one (user) role in request
            # user prompt is just a list of fewshots, this case is reduced to the no_template one
            user, status = fewshots_truncation(
                req[-1]["content"],
                target,
                instance_type,
                "no_template",
                False,
                tokenizer,
                truncation_args,
                chat_template,
                add_special_tokens,
                max_new_tokens,
                max_length - offset,
            )
        else:
            if first_system:
                system = [req[0]["content"]]
                len_system = len(system)
            else:
                len_system = 0
            user, status = fewshots_truncation(
                req[-1]["content"],
                target,
                instance_type,
                "no_template",
                False,
                tokenizer,
                truncation_args,
                chat_template,
                add_special_tokens,
                max_new_tokens,
                max_length - len_system,
            )
        if first_system:
            final = [request[0], {"role": "user", "content": user}]
        else:
            final = [{"role": "user", "content": user}]
        return final, status
    elif context_type == "multiturn":
        # for symbols truncation take into account only `content` of each dict
        if truncation_args["on"] == "symbols":
            groups = group_dicts(req, skip_system)
            final, status = fewshots_truncation(
                groups,
                target,
                instance_type,
                "no_template",
                first_system,
                tokenizer,
                truncation_args,
                chat_template,
                add_special_tokens,
                max_new_tokens,
                max_length,
            )
            result = []
            for element in final:
                for dictionary in element:
                    result.extend([dictionary])
            return result, status
        else:
            offset_max_len = max_length - max_new_tokens * int(
                instance_type == "generate_until"
            )
            # get number of fewshots (subtract system prompt and doc)
            num_fewshots = (len(req) - skip_system - 1) // 2
            if num_fewshots == 0 and skip_first:
                return request, "bad params"
            # get the number of tokens for the entire request
            tokenized_target = tokenize_sequence(
                target, add_special_tokens=False, tokenizer=tokenizer, symbols=False
            )
            tokens = (
                apply_chat_template(
                    req,
                    tokenizer,
                    chat_template,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                + tokenized_target
            )
            # fits with no truncation
            if len(tokens) <= offset_max_len:
                return request, num_fewshots
            # zero-shot, but still to long to fit into max_length
            elif len(tokens) > offset_max_len and num_fewshots == 0:
                return request, "error"
            elif len(tokens) > offset_max_len and num_fewshots == 1 and skip_first:
                return request, "bad params"
            else:
                # define the length of the system prompt (same for docs from the same task)
                if first_system:
                    system_tokens = apply_chat_template(
                        [req[0]],
                        tokenizer,
                        chat_template,
                        add_generation_prompt=False,
                        tokenize=True,
                    )
                    len_system = len(system_tokens)
                else:
                    len_system = 0

                system_and_doc = req[:skip_system] + [req[-1]]
                sys_doc_tokens = apply_chat_template(
                    system_and_doc,
                    tokenizer,
                    chat_template,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                # even if has default system prompt, len_doc takes it into account
                len_doc = len(sys_doc_tokens) - len_system
                mean_tokens = (len(tokens) - len_system - len_doc) // num_fewshots

                approx_fewshots_num = (
                    offset_max_len - len_system - len_doc - len(tokenized_target)
                ) // mean_tokens - skip_first

                groups = group_dicts(req, skip_system)

                const_parts = [[], []]
                if skip_system:
                    const_parts[0].extend(groups[0])
                if skip_first and truncation_args["side"] == "right":
                    const_parts[1].extend(groups[-2])
                    start = skip_system
                    end = -2
                elif skip_first:
                    const_parts[0].extend(groups[skip_system])
                    start = skip_system + 1
                    end = -1
                else:
                    start = skip_system
                    end = -1
                const_parts[1].extend(groups[-1])

                actual_shots = approx_fewshots_num

                if truncation_args["side"] == "right":
                    temp_result = (
                        const_parts[0]
                        + unpack_group(groups[start : start + actual_shots])
                        + const_parts[1]
                    )
                else:
                    temp_result = (
                        const_parts[0]
                        + unpack_group(groups[start:end][::-1][:actual_shots][::-1])
                        + const_parts[1]
                    )

                sum_seq = len(
                    apply_chat_template(
                        temp_result,
                        tokenizer,
                        chat_template,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                ) + len(tokenized_target)

                if sum_seq > offset_max_len:
                    for i in range(1, num_fewshots - skip_first - actual_shots):
                        if truncation_args["side"] == "right":
                            temp_result = (
                                const_parts[0]
                                + unpack_group(
                                    groups[start : start + (actual_shots - i)]
                                )
                                + const_parts[1]
                            )
                        else:
                            temp_result = (
                                const_parts[0]
                                + unpack_group(
                                    groups[start:end][::-1][: (actual_shots - i)][::-1]
                                )
                                + const_parts[1]
                            )
                        sum_seq = len(
                            apply_chat_template(
                                temp_result,
                                tokenizer,
                                chat_template,
                                tokenize=True,
                                add_generation_prompt=True,
                            )
                        ) + len(tokenized_target)
                        if sum_seq <= offset_max_len:
                            return temp_result, actual_shots - i + skip_first
                    if sum_seq > offset_max_len:
                        return request, "bad params"
                elif sum_seq < offset_max_len:
                    prev = temp_result[:]
                    for i in range(actual_shots + 1, num_fewshots - skip_first + 1):
                        if truncation_args["side"] == "right":
                            temp_result = (
                                const_parts[0]
                                + unpack_group(groups[start : start + i])
                                + const_parts[1]
                            )
                        else:
                            temp_result = (
                                const_parts[0]
                                + unpack_group(groups[start:end][::-1][:i][::-1])
                                + const_parts[1]
                            )
                        sum_seq = len(
                            apply_chat_template(
                                temp_result,
                                tokenizer,
                                chat_template,
                                tokenize=True,
                                add_generation_prompt=True,
                            )
                        ) + len(tokenized_target)
                        if sum_seq >= offset_max_len or i == num_fewshots - skip_first:
                            return prev, i
                        prev = temp_result
                else:
                    return temp_result, actual_shots + skip_first


@instance_type
@context_type
def truncate(
    request,
    target,
    instance_type,
    context_type,
    first_system,
    tokenizer,
    truncation_args,
    chat_template,
    add_special_tokens,
    max_new_tokens,
    max_length,
    **kwargs,
):
    if truncation_args["how"] == "fewshots":
        return fewshots_truncation(
            request,
            target,
            instance_type,
            context_type,
            first_system,
            tokenizer,
            truncation_args,
            chat_template,
            add_special_tokens,
            max_new_tokens,
            max_length,
        )
    elif truncation_args["how"] == "no":
        return request, "not_used"
    return request, "not_implemented"


def restore_form(request, new_query, chat_template, tokenizer):
    if isinstance(new_query, list):
        if len(new_query):
            if isinstance(new_query[0], str):
                new_query = "".join(new_query)
            elif isinstance(new_query[-1]["content"], list):
                new_query[-1]["content"] = "".join(new_query[-1]["content"])
        else:
            new_query = ""

    if not isinstance(new_query, str):
        new_query = apply_chat_template(
            new_query,
            tokenizer,
            chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
    args = request.arguments

    if len(args) == 1:
        new_pair = (new_query,)
    elif args[0] == "":
        new_pair = ("", new_query)
    else:
        new_pair = (new_query, args[1])

    request.arguments = new_pair
    return request


def truncate_and_chat_template(
    request, lm, chat_template, truncation_args, first_system
):
    if truncation_args["on"] == "symbols":
        max_len = truncation_args["max_symbols"]
        max_new = truncation_args["max_new_symbols"]
    else:
        max_len = getattr(lm, "max_length", 2048)
        max_new = getattr(lm, "max_gen_toks", 256)
    special_tokens = getattr(lm, "add_bos_token", False)
    tokenizer = getattr(lm, "tokenizer", None)
    req = request.arguments
    new_query, status = truncate(
        req,
        first_system=first_system,
        tokenizer=tokenizer,
        truncation_args=truncation_args,
        chat_template=chat_template,
        add_special_tokens=special_tokens,
        max_new_tokens=max_new,
        max_length=max_len,
    )
    processed_request = restore_form(request, new_query, chat_template, tokenizer)
    return processed_request, status
