import numpy as np
import heapq
import random
import os
import re

from utils.utils import sql2skeleton, jaccard_similarity, mask_query, sql_similarity
from utils.linking.application import mask_question_with_schema_linking
from utils.llm.chatgpt import ask_chat
from utils.data.post_process import process_duplication
from utils.embeddings.openai_embeddings import embed_dataset

from third_party.TSED import tsed_similarity
from openai import OpenAI

OPENAI_MODELS = ["text-embedding-ada-002","text-embedding-3-small", "text-embedding-3-large"]

class BasicExampleSelector(object):
    def __init__(self, data, embedding_model = None, *args, **kwargs):
        self.data = data
        self.embedding_model = embedding_model
        self.train_json = self.data.get_train_json()
        self.db_ids = [d["db_id"] for d in self.train_json]
        self.train_questions = self.data.get_train_questions()


    def get_examples(self, question, num_example, cross_domain=False):
        pass

    def domain_mask(self, candidates: list, db_id):
        cross_domain_candidates = [candidates[i] for i in range(len(self.db_ids)) if self.db_ids[i] == db_id]
        return cross_domain_candidates

    def retrieve_index(self, indexes: list, db_id):
        cross_domain_indexes = [i for i in range(len(self.db_ids)) if self.db_ids[i] == db_id]
        retrieved_indexes = [cross_domain_indexes[i] for i in indexes]
        return retrieved_indexes


class RandomExampleSelector(BasicExampleSelector):
    def __init__(self, data, embedding_model = None, *args, **kwargs):
        super().__init__(data)
        random.seed(0)

    def get_examples(self, target, num_example, cross_domain=False):
        train_json = self.train_json
        indexes = list(range(len(train_json)))
        if cross_domain:
            indexes = self.domain_mask(indexes, target["db_id"])
        selected_indexes = random.sample(indexes, num_example)
        if cross_domain:
            selected_indexes = self.retrieve_index(selected_indexes, target["db_id"])
        return [train_json[index] for index in selected_indexes]
    
    
class EuclideanDistanceSelector(BasicExampleSelector):
    def __init__(self, data, embedding_model, *args, **kwargs):
        super().__init__(data)

        from sentence_transformers import SentenceTransformer
        if self.embedding_model in OPENAI_MODELS:
            from utils.embeddings.openai_embeddings import embed_dataset
            from openai import OpenAI
            client = OpenAI()
            self.train_embeddings = embed_dataset(client, self.train_questions, self.embedding_model)
        else:
            self.bert_model = SentenceTransformer(self.embedding_model, device="cpu")    
            self.train_embeddings = self.bert_model.encode(self.train_questions)
    

    def get_examples(self, target, num_example, cross_domain=False):
        
        if self.embedding_model in OPENAI_MODELS:
            from utils.embeddings.openai_embeddings import get_embedding
            from openai import OpenAI
            client = OpenAI()
            target_embedding = get_embedding(client, target["question"], model=self.embedding_model).reshape(1, -1)
        else:
            target_embedding = self.bert_model.encode([target["question"]])


        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
    

class EuclideanDistanceQuestionMaskSelector(BasicExampleSelector):
    def __init__(self, data, embedding_model, *args, **kwargs):
        super().__init__(data)
        
        self.embedding_model = embedding_model
        self.mask_token = "<mask>"
        self.value_token = "<unk>" 

        from sentence_transformers import SentenceTransformer
        
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        
        if self.embedding_model in OPENAI_MODELS:
            from utils.embeddings.openai_embeddings import embed_dataset
            from openai import OpenAI
            client = OpenAI()
            self.train_embeddings = embed_dataset(client, train_mask_questions, self.embedding_model)
        else:
            self.bert_model = SentenceTransformer(self.embedding_model, device="cpu")    
            self.train_embeddings = self.bert_model.encode(train_mask_questions)
    

    def get_examples(self, target, num_example, cross_domain=False):
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        
        if self.embedding_model in OPENAI_MODELS:
            from utils.embeddings.openai_embeddings import get_embedding
            from openai import OpenAI
            client = OpenAI()
            target_embedding = get_embedding(client, target_mask_question[0], model=self.embedding_model).reshape(1, -1)
        else:
            target_embedding = self.bert_model.encode(target_mask_question)


        # find the most similar questions in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
            
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()

        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]


class DAILSelector(BasicExampleSelector):
    def __init__(self, data, embedding_model, *args, **kwargs):
        super().__init__(data)

        self.mask_token = "<mask>"  # the "<mask>" is the mask token of all-mpnet-base-v2
        self.value_token = "<unk>"  # the "<unk>" is the unknown token of all-mpnet-base-v2
        self.threshold = 0.85

        from sentence_transformers import SentenceTransformer
        train_mask_questions = mask_question_with_schema_linking(self.train_json, mask_tag=self.mask_token, value_tag=self.value_token)
        self.bert_model = SentenceTransformer(embedding_model, device="cpu")
        self.train_embeddings = self.bert_model.encode(train_mask_questions)

    def get_examples(self, target, num_example, cross_domain=False):
        
        target_mask_question = mask_question_with_schema_linking([target], mask_tag=self.mask_token, value_tag=self.value_token)
        target_embedding = self.bert_model.encode(target_mask_question)

        # find the most similar question in train dataset
        from sklearn.metrics.pairwise import euclidean_distances
        
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        for d, index in pairs_sorted:
            # Skeleton similarity
            if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) < self.threshold:
                continue
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        if len(top_pairs) < num_example:
            for d, index in pairs_sorted:
                # Skeleton similarity
                if jaccard_similarity(train_json[index]["pre_skeleton"], target["pre_skeleton"]) >= self.threshold:
                    continue
                top_pairs.append((index, d))
                if len(top_pairs) >= num_example:
                    break

        return [train_json[index] for (index, d) in top_pairs]

class ManualSQLSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        best_examples = []
        
        masked_gold = mask_query(target["query"])
        for example in self.train_json:
            similarity_score = float(sql_similarity(example["masked_sql"], masked_gold))
            
            if len(best_examples) < num_example:
                best_examples.append((similarity_score, example))
                best_examples.sort(key=lambda x: x[0], reverse=True)
            else:
                if similarity_score > best_examples[-1][0]:
                    best_examples[-1] = (similarity_score, example)
                    best_examples.sort(key=lambda x: x[0], reverse=True)

        # Extract the examples from the list and return them without their scores
        return [example for score, example in best_examples]



class ManualPredSQLSelector(BasicExampleSelector):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data)

    def get_examples(self, target, num_example, cross_domain=False):
        best_examples = []
        
        for example in self.train_json:
            similarity_score = float(sql_similarity(example["masked_sql"], target["pred_masked_sql"]))
            
            if len(best_examples) < num_example:
                best_examples.append((similarity_score, example))
                best_examples.sort(key=lambda x: x[0], reverse=True)
            else:
                if similarity_score > best_examples[-1][0]:
                    best_examples[-1] = (similarity_score, example)
                    best_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Extract the examples from the list and return them without their scores
        return [example for score, example in best_examples]

    
     
class EmbeddingSQLSelector(BasicExampleSelector):
    def __init__(self, data, embedding_model, *args, **kwargs):
        super().__init__(data)
        
        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(embedding_model, device="cpu")
        
        train_mask_sqls = [item["masked_sql"] for item in self.train_json]
        
        self.train_sql_embeddings = self.bert_model.encode(train_mask_sqls)
        
    def get_examples(self, target, num_example, cross_domain=False):
        
        target_embedding = self.bert_model.encode([mask_query(target["query"])])
        
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_sql_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        
        for d, index in pairs_sorted:
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]


class EmbeddingPredSQLSelector(BasicExampleSelector):
    def __init__(self, data, embedding_model, *args, **kwargs):
        super().__init__(data)
        
        from sentence_transformers import SentenceTransformer
        self.bert_model = SentenceTransformer(embedding_model, device="cpu")
        
        train_mask_sqls = [item["masked_sql"] for item in self.train_json]
        
        self.train_sql_embeddings = self.bert_model.encode(train_mask_sqls)
        
        
    def get_examples(self, target, num_example, cross_domain=False):
        
        target_embedding = self.bert_model.encode([target["pred_masked_sql"]])
        
        from sklearn.metrics.pairwise import euclidean_distances
        distances = np.squeeze(euclidean_distances(target_embedding, self.train_sql_embeddings)).tolist()
        pairs = [(distance, index) for distance, index in zip(distances, range(len(distances)))]

        train_json = self.train_json
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        top_pairs = list()
        
        for d, index in pairs_sorted:
            top_pairs.append((index, d))
            if len(top_pairs) >= num_example:
                break

        return [train_json[index] for (index, d) in top_pairs]
