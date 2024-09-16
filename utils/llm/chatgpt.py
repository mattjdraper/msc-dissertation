import json.decoder

import openai
from utils.parameters import LLM
import time


def ask_completion(client, model, batch, temperature):
    response = client.chat.completions.create(
        model=model,
        prompt=batch,
        temperature=temperature,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[";"]
    )
    response_clean = [_["text"] for _ in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )
    

def ask_chat(client, model, messages: list, temperature, n):
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=200,
        n=n
    )
    response_clean = [choice.message.content for choice in response.choices]
    response_data = {
        "response": response_clean,
        "completion_tokens": response.usage.completion_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens
    }
    if n == 1:
        response_data["response"] = response_clean[0]
        
    return response_data


def ask_llm(client, model: str, batch: list, temperature: float, n:int):
    n_repeat = 0
    while True:
        try:
            if model in LLM.TASK_COMPLETIONS:
                # TODO: self-consistency in this mode
                assert n == 1
                response = ask_completion(client, model, batch, temperature)
                
            elif model in LLM.TASK_CHAT:
                # batch size must be 1
                assert len(batch) == 1, "batch must be 1 in this mode"
                messages = [{"role": "user", "content": batch[0]}]
                response = ask_chat(client, model, messages, temperature, n)
      
                response['response'] = [response['response']]
            break
        
        except openai.error.RateLimitError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for RateLimitError", end="\n")
            time.sleep(1)
            continue
        except json.decoder.JSONDecodeError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
            time.sleep(1)
            continue
        except Exception as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            time.sleep(1)
            continue

    return response

