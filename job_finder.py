#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:22:30 2023

@author: deni-kun
"""

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def calculate_bert_embedding(text):
    # Tokenize input
    input_ids = tokenizer.encode(text, add_special_tokens=True) 

    # Create attention masks
    attention_mask = [1] * len(input_ids)

    # Pad token list to max length of model (512 for BERT)
    padding_length = 512 - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    # Convert to PyTorch tensors
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)

    # Get BERT embeddings
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    # Get sentence embedding from BERT output (mean of all token vectors)
    sentence_embedding = torch.mean(last_hidden_states[0], dim=1).numpy()

    return sentence_embedding.squeeze()  # Ensure the embedding is 1-D


job_description = st.text_area('Enter the job offer description')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # Perform necessary cleaning such as removing NaNs, dealing with duplicates, etc.
    
    st.dataframe(data=data)

    
    # Concatenate job description with each person's skills
    data['combined_skills'] = data['skills'].apply(lambda x: ' '.join([job_description, x]))
    data['bert_embeddings'] = data['combined_skills'].apply(calculate_bert_embedding)
    
    # Apply TF-IDF
    #vectorizer = TfidfVectorizer()
    #data['tfidf'] = list(vectorizer.fit_transform(data['combined_skills']).toarray())

    # Calculate similarity with the job description
    job_description_embedding = calculate_bert_embedding(job_description)
    data['similarity'] = data['bert_embeddings'].apply(lambda x: 1 - cosine(x, job_description_embedding))
    
    # Sort by similarity
    data = data.sort_values('similarity', ascending=False)
    
    st.header('Similarity scores')
    
    head_num = st.selectbox('How many people are needed?', (1, 2, 3, 4, 5))
    
    #st.dataframe(data=data)
    
    st.write(data[['name', 'similarity']].head(head_num))

