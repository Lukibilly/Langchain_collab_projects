import openai
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
import os
import numpy

openai.api_key = os.environ["OPENAI_API_KEY"]

llm = OpenAI(model_name="text-davinci-003",temperature=0.5, max_tokens=100, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)

template = """
The following are examples of something a Person would say. Generate a new example of something the Person would say.
{Examples}
Result: 
"""
prompt = PromptTemplate(
    input_variables=["Examples"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

Benny = """
Example1: DnD is cool.
Example2: Experimental Physics is cool.
"""

Puhl = """
Example1: Avatar is the best show ever.
Example2: Computational Physics is cool.
"""

Janik = """
Example1: Playinmg the drums is fun.
Example2: Particle Physics is cool.
"""


inputs = [{"Examples": Benny},
          {"Examples": Puhl},
          {"Examples": Janik}]

generations = chain.generate(inputs).generations

result_Benny = generations[0][0]
result_Puhl = generations[1][0]
result_Janik = generations[2][0]


#text2 = result2.text
print('Benny: ',result_Benny.text)
print('Puhl: ',result_Puhl.text)
print('Janik: ',result_Janik.text)
#print(text2)


# You could simulate conversations between people using their interests (or maybe posts, following people, etc.) and link people that are likely to connect well.
