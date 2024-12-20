from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline

class ContentGenerator:
    def __init__(self, model_name="nisten/Biggie-SmoLlm-0.15B-Base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=100
        )
        
        # Create LLM
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.chain = self._create_chain()
    
    def _create_chain(self):
        prompt = PromptTemplate(
            template="Write a creative story based on: {prompt}",
            input_variables=["prompt"]
        )
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def generate(self, prompt):
        """Generate creative content based on the prompt"""
        return self.chain.run(prompt=prompt)
