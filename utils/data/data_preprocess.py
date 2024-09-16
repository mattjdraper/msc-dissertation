import platform
import sys
import json
import os
import subprocess
from pathlib import Path
import sqlite3
from tqdm import tqdm
import shutil
import time

from utils.linking.linking_process import SpiderEncoderV2Preproc
from utils.embeddings.pretrained_embeddings import GloVe
from utils.data.spider_utils import load_tables

def schema_linking_producer(test, train, table, db, dataset_dir, data_type, compute_cv_link=True):

    # load data
    test_data = json.load(open(os.path.join(dataset_dir, test)))
    train_data = json.load(open(os.path.join(dataset_dir, train)))

    # load schemas
    schemas, _ = load_tables([os.path.join(dataset_dir, table)])

    if data_type == "spider":
        # Backup in-memory copies of all the DBs and create the live connections
        # BIRD Dataset too large to be stored in local RAM memory
        for db_id, schema in tqdm(schemas.items(), desc="DB connections"):
            sqlite_path = Path(dataset_dir) / db / db_id / f"{db_id}.sqlite"
            source: sqlite3.Connection
            print("sqlite_path", sqlite_path)
            with sqlite3.connect(str(sqlite_path)) as source:
                dest = sqlite3.connect(':memory:')
                dest.row_factory = sqlite3.Row
                source.backup(dest)
            schema.connection = dest

    word_emb = GloVe("42B", lemmatize=True)
    linking_processor = SpiderEncoderV2Preproc(dataset_dir,
            min_freq=4,
            max_count=5000,
            include_table_name_in_column=False,
            word_emb=word_emb,
            fix_issue_16_primary_keys=True,
            compute_sc_link=True,
            compute_cv_link=compute_cv_link)

    # build schema-linking
    for data, section in zip([test_data, train_data],['test', 'train']):
        for item in tqdm(data, desc=f"{section} section linking"):
            db_id = item["db_id"]
            schema = schemas[db_id]
            to_add, validation_info = linking_processor.validate_item(item, schema, section)
            if to_add:
                linking_processor.add_item(item, schema, section, validation_info)

    # save
    linking_processor.save()


def bird_pre_process(bird_dir, with_evidence=False):
    
    new_db_path = os.path.join(bird_dir, "databases")
    
    if not os.path.exists(new_db_path):
        os.makedirs(new_db_path)  # Ensure the target directory exists
        
    
    if (os.path.exists(new_db_path) and not os.listdir(new_db_path)):
        train_db_path = os.path.join(bird_dir, os.path.join('train','train_databases'))
        dev_db_path = os.path.join(bird_dir, os.path.join('dev','dev_databases'))
        
        # Copy each subdirectory from train and dev databases to the new database directory
        print("Copying databases from train and dev directories to the new databases directory...")
        print("This may take a while...")
        for directory in os.listdir(train_db_path):
            full_dir_path = os.path.join(train_db_path, directory)
            if os.path.isdir(full_dir_path):
                shutil.copytree(full_dir_path, os.path.join(new_db_path, directory))
        
        for directory in os.listdir(dev_db_path):
            full_dir_path = os.path.join(dev_db_path, directory)
            if os.path.isdir(full_dir_path):
                shutil.copytree(full_dir_path, os.path.join(new_db_path, os.path.basename(directory)))

    def json_preprocess(data_jsons):
        new_datas = []
        for i, data_json in enumerate(data_jsons, start=1):
            ### Append the evidence to the question
            if with_evidence and len(data_json["evidence"]) > 0:
                data_json['question'] = (data_json['question'] + " " + data_json["evidence"]).strip()
            question = data_json['question']
            tokens = []
            for token in question.split(' '):
                if len(token) == 0:
                    continue
                if token[-1] in ['?', '.', ':', ';', ','] and len(token) > 1:
                    tokens.extend([token[:-1], token[-1:]])
                else:
                    tokens.append(token)
            data_json['question_toks'] = tokens
            data_json['query'] = data_json['SQL']
            data_json['ex_id'] = i
            new_datas.append(data_json)
        return new_datas

    output_dev = 'dev.json'
    output_train = 'train.json'
    with open(os.path.join(bird_dir, 'dev', 'dev.json')) as f:
        data_jsons = json.load(f)
        wf = open(os.path.join(bird_dir, output_dev), 'w')
        json.dump(json_preprocess(data_jsons), wf, indent=4)
    with open(os.path.join(bird_dir, 'train','train.json')) as f:
        data_jsons = json.load(f)
        wf = open(os.path.join(bird_dir, output_train), 'w')
        json.dump(json_preprocess(data_jsons), wf, indent=4)
   
    # Copy dev.sql to bird_dir
    shutil.copy(os.path.join(bird_dir, 'dev','dev.sql'), bird_dir)

    # Copy train_gold.sql to bird_dir
    shutil.copy(os.path.join(bird_dir, 'train','train_gold.sql'), bird_dir)
    tables = []
    with open(os.path.join(bird_dir, 'dev','dev_tables.json')) as f:
        tables.extend(json.load(f))
    with open(os.path.join(bird_dir, 'train','train_tables.json')) as f:
        tables.extend(json.load(f))
    with open(os.path.join(bird_dir, 'tables.json'), 'w') as f:
        json.dump(tables, f, indent=4)
        


if __name__ == '__main__':
    
    print("\nNOTE: This script will take a while to run. Port 9000 must be empty before running this script.")
    print("This is to ensure the third-party CoreNLP server script can connect.")
    
    system = platform.system()

    if system == "Windows":
        print("\nIf you have a server running on port 9000, please stop it before running this script.")
        print("Notably, any active Jupyter Notebook kernels should be killed.")
        print("This can be achieved by running the following commands in the terminal (Windows):")
        print("1. Check which process is using port 9000:")
        print("   netstat -ano | findstr :9000")
        print("2. Kill the process using the PID from the previous command:")
        print("   taskkill /PID <PID> /F")
        print()
    
    elif system == "Darwin": 
        print("\nIf you have a server running on port 9000, please stop it before running this script.")
        print("Notably, any active Jupyter Notebook kernels should be killed.")
        print("This can be achieved by running the following commands in the terminal (macOS):")
        print("1. Check which process is using port 9000:")
        print("   lsof -i :9000")
        print("2. Kill the process using the PID from the previous command:")
        print("   kill -9 <PID>")
        print()
        
    time.sleep(5)
    print("Preprocessing Spider datasets")
    print()
    time.sleep(2)

    spider_dir = os.path.join("benchmarks", "spider")
    split1 = "train_spider.json"
    split2 = "train_others.json"
    total_train = []
    # Merge two training split of Spider
    for item in json.load(open(os.path.join(spider_dir, split1))):
        total_train.append(item)
    for item in json.load(open(os.path.join(spider_dir, split2))):
        total_train.append(item)
    # s2593814 script: add id's to each example, improve format of json.dump.
    for i, item in enumerate(total_train, start=1):
        item['ex_id'] = i
    with open(os.path.join(spider_dir, 'train_spider_and_others.json'), 'w') as f:
        json.dump(total_train, f, indent=4)    
    # Schema-linking between questions and databases for Spider
    spider_dev = "dev.json"
    spider_train = 'train_spider_and_others.json'
    spider_table = 'tables.json'
    spider_db = 'databases'
    schema_linking_producer(spider_dev, spider_train, spider_table, spider_db, spider_dir, data_type = "spider")
    
    
    time.sleep(5)
    print("Preprocessing BIRD datasets")
    print()
    time.sleep(2)
        
    bird_dir = os.path.join("benchmarks", "bird")
    
    # Schema-linking for bird with evidence
    bird_pre_process(bird_dir, with_evidence=True)
    bird_dev = 'dev.json'
    bird_train = 'train.json'
    bird_table = 'tables.json'
    bird_db = 'databases'
    # Do NOT compute the cv_link since it is time-consuming for the huge database of BIRD
    schema_linking_producer(bird_dev, bird_train, bird_table, bird_db, bird_dir, data_type = "bird", compute_cv_link=False)
    
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    print()
    print("Preprocessing completed! You can now run the experiment notebooks.")
