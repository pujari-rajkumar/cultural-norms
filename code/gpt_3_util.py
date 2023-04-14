from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential
)
import json
import openai
from datetime import datetime
from pprint import pprint

_config = {"temperature": 0.9,
           "max_tokens": 1024,
           "top_p": 1.0,
           "frequency_penalty": 0.0,
           "presence_penalty": 0.6}


@retry(wait=wait_random_exponential(min=1, max=60),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(openai.error.RateLimitError))
def _get_response(prompt: str, model_name: str = "text-davinci-003", debug_log_path: str = None) -> dict:
    result = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=_config['temperature'],
        max_tokens=_config['max_tokens'],
        top_p=_config['top_p'],
        frequency_penalty=_config['frequency_penalty'],
        presence_penalty=_config['presence_penalty']
    )
    if debug_log_path is not None:
        with open(datetime.now().strftime(debug_log_path + model_name + " _log_%Y_%m_%d_%H_%M_%S_%f.json"), "w") as fp:
            json.dump(result, fp)
    return result


@retry(wait=wait_random_exponential(min=1, max=60),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(openai.error.RateLimitError))
def _get_response_chat(message: list, model_name: str = "gpt-3.5-turbo", debug_log_path: str = None) -> dict:
    result = openai.ChatCompletion.create(
        model=model_name,
        messages=message,
        temperature=_config['temperature'],
        max_tokens=_config['max_tokens'],
        top_p=_config['top_p'],
        frequency_penalty=_config['frequency_penalty'],
        presence_penalty=_config['presence_penalty']
    )
    if debug_log_path is not None:
        with open(datetime.now().strftime(debug_log_path + model_name + "_log_%Y_%m_%d_%H_%M_%S_%f.json"), "w") as fp:
            json.dump(result, fp)
    return result


def _append_question(prompt: str, question: str, delimiter: str = '\n\n') -> str:
    return (prompt + delimiter + question).strip()


def _append_question_chat(context: list, question: str):
    context.append({"role": "user", "content": question})


def _append_response_chat(context: list, response: str):
    context.append({"role": "assistant", "content": response})


def _append_sequence(prompt: str, seq: str) -> str:
    return seq + prompt


def _extract_raw_result(raw: dict) -> str:
    return raw['choices'][0]['text'].strip()


def _extract_raw_result_chat(raw: dict) -> str:
    return raw['choices'][0]['message']['content'].strip()


def generate_explanation(conv: str,
                         questions: list,
                         model_name: str = "gpt-3.5-turbo",
                         verbose: bool = False,
                         debug_log: str = None) -> dict:
    init_conv = _append_question(conv, questions[0])
    output_dict = dict()
    output_dict['dialogue'] = conv
    curr_response = _extract_raw_result(_get_response(init_conv, model_name, debug_log))

    full_context = init_conv
    full_context += '\n' + curr_response

    output_dict[questions[0]] = curr_response
    for q in questions[1:]:
        curr_response = _extract_raw_result(_get_response(_append_question(full_context, q), model_name, debug_log))
        output_dict[q] = curr_response
        full_context = _append_question(full_context, q)
        full_context += '\n' + curr_response
    if verbose:
        print(full_context)
        print()
    return output_dict


def generate_explanation_chat(conv: str,
                              questions: list,
                              model_name: str = "gpt-3.5-turbo",
                              verbose: bool = False,
                              task_desc: str = "",
                              debug_log: str = None) -> dict:
    context = list()
    if task_desc != "":
        context.append({"role": "system", "content": task_desc})
    _append_question_chat(context, _append_question(conv, questions[0]))
    output_dict = dict()
    output_dict['dialogue'] = conv
    curr_response = _extract_raw_result_chat(_get_response_chat(context, model_name, debug_log))
    _append_response_chat(context, curr_response)

    output_dict[questions[0]] = curr_response
    for q in questions[1:]:
        _append_question_chat(context, q)
        curr_response = _extract_raw_result_chat(_get_response_chat(context, model_name, debug_log))
        output_dict[q] = curr_response
        _append_response_chat(context, curr_response)

    if verbose:
        pprint(context)
        print()
    return output_dict


def init(api_key: str, conf: dict = None):
    global _config
    openai.api_key = api_key
    if conf:
        for k in list(conf.keys()):
            if k in _config:
                _config[k] = conf[k]
