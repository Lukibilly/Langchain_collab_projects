import openai
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import numpy

openai.api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(model_name="text-davinci-003",temperature=0, max_tokens=100, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)



template = """
Blabla
{Input}
Blabla
"""
prompt = PromptTemplate(
    input_variables=["Input"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

input1 = """
yepyep
"""

result = chain.run(input1)

print(result)
