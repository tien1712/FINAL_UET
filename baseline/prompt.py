from asyncore import poll3
from poplib import POP3_PORT
import sys
import pandas as pd
from langchain.prompts import ChatPromptTemplate
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FAISS.retrieval import retrieval

# Định nghĩa template với các placeholder
template = """
{p1}
{p2}
{p3}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def prompt(row):
    # p1: Task description
    p1 = "You are a transportation behavior expert that predicts trip mode (either 'Public transports (train, bus, tram, etc.)', 'Private modes (car, motorbike, etc.)', 'Soft modes (bike, walk, etc.)'). Based on the provided trip details, their previous trip choices and similar past trips, what is the most likely trip mode? Only output one of: [Public transports, Private modes, Soft modes]. Do not provide any explanation or additional text, just output the mode name."
    
    # p2: Input information
    p2 = "Trip details: \n" + row["INFOR"]
    
    # p3: Retrieval information
    situations, examples = retrieval(row["INFOR"], row["ID"])
    
    formatted_results = "The person's previous choices:\n"
    formatted_results += "".join(situations)
    formatted_results += "Read similar trips from others to get a general understanding. The similar trips:\n"
    formatted_results += "".join(examples)
    
    p3 = formatted_results
    
    # p4: Output format instruction
    #p4 = "Please infer what is the mostly likely travel mode that the person will choose. Organize your answer in a JSON object with two keys: 'prediction' (the predicted travel mode) and 'reason' (explanation that supports your inference)."
    
    
    final_prompt = prompt_template.invoke({
        "p1": p1,
        "p2": p2,
        "p3": p3
    })
    
    return final_prompt
    
# test prompt
if __name__ == "__main__":
    df = pd.read_csv("data/Optima/test.csv")
    row = df.iloc[3]
    prompt_value = prompt(row)
    result = str(prompt_value)  # convert to string

    print(result)
    
    