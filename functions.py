# imports
import os # dealing with directories
import pypdf # convert pdf to text
from sentence_transformers import SentenceTransformer # embedding model
from transformers import AutoTokenizer, AutoModelForCausalLM # language model
import torch # helper for hugging face tools
import numpy as np # handling vectores
import faiss # vectore database

# find uploaded document path
def documents_path(directory='/content/'):
  doc_list = os.listdir('/content/')
  doc_list.remove('.config')
  doc_list.remove('sample_data')
  return doc_list

# convert pdf to text
def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# saving all texts as 1 text file
def text_of_docs(docs_path):
  full_text = ""
  for p in docs_path:
    full_text += pdf_to_text("/content/" + p)
  return full_text

# chunk documents
def chunk_text(text, n, overlap):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks

# embedding chunks
def embedding_chunks(chunks, model_name = 'nomic-ai/modernbert-embed-base'):
  embedding_model = SentenceTransformer(model_name)
  embeddings = embedding_model.encode(chunks)
  return embeddings

# vectore database
def vector_db(embeddings, chunks):
  # Create a FAISS index
  index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 is Euclidean distance

  # Add our vectors to the index
  index.add(embeddings.astype(np.float32))  # FAISS requires float32

  # Create a mapping from index position to document chunk for retrieval
  index_to_doc_chunk = {i: doc for i, doc in enumerate(chunks)}

  return index, index_to_doc_chunk

# LLM model
def LLM(model_name = 'Qwen/Qwen2.5-7B-Instruct'):
  model = AutoModelForCausalLM.from_pretrained(
          model_name,
          torch_dtype="auto",
          device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  return model, tokenizer

# generate prompt according to context and query
def query2prompt(query, index, index_mapping, top_k=10):
  # Step 1: Convert query to embedding
  query_embedding = embedding_chunks([query])
  query_embedding = query_embedding.astype(np.float32)  # Convert to float32 for FAISS

  # Step 2: Search for similar documents
  distances, indices = index.search(query_embedding, 10)

  # Step 3: Retrieve the actual document chunks
  retrieved_docs = [index_mapping[idx] for idx in indices[0]]

  # Create context from retrieved documents
  context = "nn".join(retrieved_docs)

  prompt = f"""
  Context:
     {context}
   <|user|>
   {query}
   <|assistant|>
   """

  return prompt

# generate output by prompt
def generate_message(prompt):
  messages = [
      {"role": "system", "content": '''You are a helpful AI assistant. Answer the question based only on the provided context.
      If you don't know the answer based on the context, say "I don't have enough information to answer this question."'''},
      {"role": "user", "content": prompt}
  ]

  text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

  generated_ids = model.generate(
          **model_inputs,
          max_new_tokens=512
  )

  generated_ids = [
      output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]
  
  response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

  return response
