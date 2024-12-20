from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class ContentGenerator:
    def __init__(self, model_name="google/flan-t5-large", knowledge_base_path="data/training_data"):
        self.embeddings = HuggingFaceEmbeddings()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.vector_store = None
        self.knowledge_base_path = knowledge_base_path
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        """Initialize the vector store with knowledge base documents"""
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path)
            return

        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # Load documents from knowledge base
        for filename in os.listdir(self.knowledge_base_path):
            if filename.endswith('.txt'):
                with open(os.path.join(self.knowledge_base_path, filename), 'r', encoding='utf-8') as f:
                    text = f.read()
                    chunks = text_splitter.split_text(text)
                    documents.extend(chunks)

        if documents:
            self.vector_store = FAISS.from_texts(documents, self.embeddings)

    def generate(self, prompt, max_length=500):
        """Generate content using RAG if available, fallback to base model"""
        if self.vector_store:
            # Use RAG for generation
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.7}
                ),
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = qa_chain({"query": prompt})
            return result["result"]
        else:
            # Fallback to base model
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def add_to_knowledge_base(self, text, filename=None):
        """Add new text to the knowledge base"""
        if filename is None:
            filename = f"document_{len(os.listdir(self.knowledge_base_path))}.txt"
        
        file_path = os.path.join(self.knowledge_base_path, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Reinitialize knowledge base
        self.initialize_knowledge_base()
