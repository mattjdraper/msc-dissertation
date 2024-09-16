import argparse
import os
import json

import openai
from openai import OpenAI
from openai import RateLimitError, Timeout, APIError, APIConnectionError, OpenAIError
from tqdm import tqdm

from utils.llm.chatgpt import ask_llm
from utils.parameters import LLM
from torch.utils.data import DataLoader
from utils.utils import gpt4_format

from utils.data.post_process import process_duplication, get_sqls

QUESTION_FILE = "questions.json"

def run_llm(QUESTION_FILE, 
            OUTPUT_FILE,
            openai_api_key, 
            openai_group_id, 
            model = LLM.GPT_35_TURBO, 
            start_index = 0, 
            end_index = 1000000, 
            temperature = 0, 
            n = 1, 
            db_dir = "benchmarks/spider/database"):
    
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['OPENAI_ORG_ID'] = openai_group_id

    questions_json = json.load(open(QUESTION_FILE, "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    # init openai api
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))


    if start_index == 0:
        mode = "w"
    else:
        mode = "a"

    question_loader = DataLoader(questions, batch_size= 1, shuffle=False, drop_last=False)

    token_count = 0
    with open(OUTPUT_FILE, mode) as f:
        for i, batch in enumerate(tqdm(question_loader)):
            if i < start_index:
                continue
            if i >= end_index:
                break
            try:
                res = ask_llm(client, model, batch, temperature, n)
                #print(res)
            except OpenAIError:
                print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                res = ""

            # parse result
            token_count += res["total_tokens"]
            if n == 1 and model != "gpt-4-turbo" and model != "gpt-4o":
                for sql in res["response"]:
                    # remove \n and extra spaces
                    sql = " ".join(sql.replace("\n", " ").split())
                    sql = process_duplication(sql)
                    if sql.startswith("SELECT"):
                        f.write(sql + "\n")
                    elif sql.startswith(" "):
                        f.write("SELECT" + sql + "\n")
                    else:
                        f.write("SELECT " + sql + "\n")
            elif n == 1 and (model == "gpt-4-turbo" or model == "gpt-4o"):
                sql = gpt4_format(res["response"][0])
                f.write(sql + "\n")
            else:
                results = []
                cur_db_ids = db_ids[i * 1: i * 1 + len(batch)]
                for sqls, db_id in zip(res["response"], cur_db_ids):
                    processed_sqls = []
                    for sql in sqls:
                        sql = " ".join(sql.replace("\n", " ").split())
                        sql = process_duplication(sql)
                        if sql.startswith("SELECT"):
                            pass
                        elif sql.startswith(" "):
                            sql = "SELECT" + sql
                        else:
                            sql = "SELECT " + sql
                        processed_sqls.append(sql)
                    result = {
                        'db_id': db_id,
                        'p_sqls': processed_sqls
                    }
                    final_sqls = get_sqls([result], n, db_dir)

                    for sql in final_sqls:
                        f.write(sql + "\n")

