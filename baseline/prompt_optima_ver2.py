from asyncore import poll3
from poplib import POP3_PORT
import sys
import pandas as pd
from langchain.prompts import ChatPromptTemplate
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FAISS.retrieval_optima import retrieval

# Định nghĩa template với các placeholder
template = """
{p1}
{p2}
{p3}
{p4}
{p5}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def prompt(row):
    # p1: Task description
    p1 = "You are a transportation behavior expert that predicts trip mode (either 'Public transports (train, bus, tram, etc.)', 'Private modes (car, motorbike, etc.)', 'Soft modes (bike, walk, etc.)'). Based on the provided trip details, their previous trip choices and similar past trips to solve the task. "
    p1 += "Task: What is the most likely trip mode? "
    #Note the '...' places are where information is missing.   
    
    #p2: Features description
    p2 = "Features:"
    p2 += "- TimePT: Total duration of the travel performed in public transport (in minutes)"
    p2 += "- TimeCar: Total duration of the travel performed using the car (in minutes)"
    p2 += "- MarginalCostPT: Total cost in public transport taking into account the possible discounts"
    p2 += "- CostCarCHF: Total cost of a travel performed using the car (in Swiss francs)"
    p2 += "- distance km: Total distance (in kilometers)"
    p2 += "- age: Age of the individual"
    p2 += "- NbChild: Number of kids in the household"
    p2 += "- NbCar: Number of cars the household"
    p2 += "- NbMoto: Number of motorbikes in the household" 
    p2 += "- NbBicy: Number of bikes in the household"
    p2 += "- Gender: Gender of the individual"
    p2 += "- OccupStat: Occupational status of the individual"
    
    # p3: Input information
    p3 = "Trip details: " + row["INFOR"]
    
    # p4: Retrieval information
    situations, examples = retrieval(row["INFOR"], row["ID"])
    
    formatted_results = "The person's previous choices: "
    if situations:
        formatted_results += "".join(situations)
    else:
        formatted_results += "No previous choices information. "
    formatted_results += "The similar trips from others: "
    if examples:
        formatted_results += "".join(examples)
    else:
        formatted_results += "No similar trips information."
    
    p4 = formatted_results
    
    #p5: Processing logic
    p5 = "Let's first understand the problem and solve the problem step by step. "
    p5 += "Step 1. Analyze the causal relationship or tendency between each feature and task description based on general knowledge and common sense within a short sentence. "
    p5 += "Step 2. Based on the above examples, previous choices and Step 1's results, infer 10 different conditions when predicting the trip mode. The condition should make sense, well match examples and previous choices. "
    p5 += "Step 3. Based on the above conditions, infer the most likely trip mode that the person will choose. "
    
    #p5: Output format instruction
    p5 = "Only output one of: [Public transports, Private modes, Soft modes]. "
    
    final_prompt = prompt_template.invoke({
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5
    })
    
    return final_prompt
    
# test prompt
if __name__ == "__main__":
    df = pd.read_csv("data/Optima/test1.csv")
    row = df.iloc[1]
    prompt_value = prompt(row)
    result = str(prompt_value)  # convert to string

    print(result)
    
    