import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

def load_json(dir):
    with open(dir, 'r') as j:
        contents = json.loads(j.read())
    return contents


def result_callback(result, exec_result=None):  # Step 4: Modify to accept the shared list
    if exec_result is not None:
        exec_result.append(result)
    print(f"Result received for query pair {result['sql_idx']}:", result)

def execute_sql(predicted_sql,ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res



def execute_model(predicted_sql,ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    # print(result)
    # result = str(set([ret[0] for ret in result]))
    result = {'sql_idx': idx, 'res': res}
    # print(result)
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dev'):
    clean_sqls = []
    db_path_list = []
    if mode == 'gpt':
        sql_data = json.load(open(sql_path, 'r'))
        for idx, sql_str in sql_data.items():
            if type(sql_str) == str:
                sql, db_name = sql_str.split('\t----- bird -----\t')
            else:
                sql, db_name = " ", "financial"
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        sqls = open(sql_path)
        sql_txt = sqls.readlines()
        # sql_txt = [sql.split('\t')[0] for sql in sql_txt]
        for idx, sql_str in enumerate(sql_txt):
            sql, db_name = sql_str.strip().split('\t')
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0, exec_result=None):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=lambda result: result_callback(result, exec_result))
    pool.close()
    pool.join()


def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc_by_diff(exec_results,diff_json_path):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    simple_results, moderate_results, challenging_results = [], [], []

    for i,content in enumerate(contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])

        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])

        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results)
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results)
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results)
    all_acc = sum(results)/num_queries
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists



def print_data(score_lists,count_lists):
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))

    print('======================================    ACCURACY    =====================================')
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))

def reformat_responses(prompts_path, responses_path, output_path):
    # Load the responses from the responses file
    with open(responses_path, 'r') as file:
        responses = file.readlines()
    
    # Load the prompts to get the db_id for each response
    with open(prompts_path, 'r') as file:
        prompts = json.load(file)
    
    # Initialize an empty dictionary to store the reformatted responses
    reformatted_responses = {}
    
    # Iterate through the responses and reformat them
    for i, response in enumerate(responses):
        # Trim newline characters and other potential whitespace
        response = response.strip()
        
        # Append the necessary suffix using the db_id from the prompts
        db_id = prompts['questions'][i]['db_id']
        reformatted_response = f"{response}\t----- bird -----\t{db_id}"
        
        # Add the reformatted response to the dictionary
        reformatted_responses[str(i)] = reformatted_response
    
    # Write the reformatted responses to the output file
    with open(output_path, 'w') as file:
        json.dump(reformatted_responses, file, indent=4)


def evaluate_bird(gold, pred, db, data_mode='dev', meta_time_out=10, num_cpus=1, mode_gt='gt', mode_predict='gpt', difficulty='simple', diff_json_path=''):
    manager = mp.Manager()
    exec_result = manager.list()  # Step 2: Create a shared list

    pred_queries, db_paths = package_sqls(pred, db, mode=mode_predict, data_mode=data_mode)
    gt_queries, db_paths_gt = package_sqls(gold, db, mode='gt', data_mode=data_mode)

    query_pairs = list(zip(pred_queries, gt_queries))
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=num_cpus, meta_time_out=meta_time_out, exec_result=exec_result)  # Step 3: Pass the shared list
    exec_result = sort_results(list(exec_result))  # Convert back to a regular list before sorting

    print('start calculate')
    simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(exec_result, diff_json_path)
    score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
    print_data(score_lists, count_lists)
    print('===========================================================================================')
    print("Finished evaluation")
    
    results = [res['res'] for res in exec_result]
    
    return results
    