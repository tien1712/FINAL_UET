from asyncore import poll3
from poplib import POP3_PORT
import sys
import pandas as pd
from langchain.prompts import ChatPromptTemplate
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Định nghĩa template với các placeholder
template = """
{p1}
{p2}
{p3}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def prompt(row):
    # p1: Task description
    p1 = "You are a transportation behavior expert that thinking about the person's transportation behavior. Based on behavioral survey questions. The task is to infer 3 conditions that describe the person's transportation behavior when choosing 'Public transports (train, bus, tram, etc.)', 'Private modes (car, motorbike, etc.)', 'Soft modes (bike, walk, etc.)'). The output is a paragraph containing information about three conditions in the format: 'That person uses Public transports if: [condition], Use Private modes if: [condition], Use Soft modes if: [condition]'. The paragraph must be clear and in the correct format. "
    #p1 += "There are 5 levels of responses: 1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree, 6 = not applicable. (note: -1 = missing value)"
    
     #p2: question description
    p2 = "There are 5 levels of responses: 1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree, 6 = not applicable. (note: -1 = missing value). "
    p2 += "Questions:"
    p2 += "- Mobil01: I use my current mean of transport mode because I have no alternative.\n"
    p2 += "- Mobil02: In general, for my activities, I always have a usual mean of transport.\n"
    p2 += "- Mobil03: With my car I can go wherever and whenever.\n"
    p2 += "- Mobil04: When I take the car I know I will be on time.\n"
    p2 += "- Mobil05: I do not like looking for a parking place.\n"
    p2 += "- Mobil06: I do not like changing the mean of transport when I am traveling.\n"
    p2 += "- Mobil07: If I use public transportation I have to cancel certain activities I would have done if I had taken the car."
    
    # p3: Input information
    p3 = "Responses to the questions: " 
    p3 += "Mobil01: " + str(row["Mobil06"])
    p3 += ". Mobil02: " + str(row["Mobil07"]) 
    p3 += ". Mobil03: " + str(row["Mobil13"]) 
    p3 += ". Mobil04: " + str(row["Mobil14"]) 
    p3 += ". Mobil05: " + str(row["Mobil15"]) 
    p3 += ". Mobil06: " + str(row["Mobil16"]) 
    p3 += ". Mobil07: " + str(row["Mobil17"])
    

    
    final_prompt = prompt_template.invoke({
        "p1": p1,
        "p2": p2,
        "p3": p3
    })
    
    return final_prompt
    
# test prompt
if __name__ == "__main__":
    df = pd.read_csv("data/Optima/paperdata/behavior_clean.csv")
    row = df.iloc[17]
    prompt_value = prompt(row)
    result = str(prompt_value)  # convert to string

    print(result)
    
    