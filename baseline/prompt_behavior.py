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
{p4}
"""
prompt_template = ChatPromptTemplate.from_template(template)

def prompt(row):
    # p1: Task description
    p1 = "You are a transportation behavior expert that thinking about the person's transportation behavior. Based on behavioral survey questions. The task is to infer the person's transportation behavior. "
    p1 += "There are 5 levels of responses: 1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree, 6 = not applicable. (note: -1 = missing value)"
    
     #p2: question description
    p2 = "Questions:"
    p2 += "- Mobil01: My trip is a useful transition between home and work.\n"
    p2 += "- Mobil02: The trip I must do interferes with other things I would like to do.\n"
    p2 += "- Mobil03: I use the time of my trip in a productive way.\n"
    p2 += "- Mobil04: Being stuck in traffic bores me.\n"
    p2 += "- Mobil05: I reconsider frequently my mode choice.\n"
    p2 += "- Mobil06: I use my current mean of transport mode because I have no alternative.\n"
    p2 += "- Mobil07: In general, for my activities, I always have a usual mean of transport.\n"
    p2 += "- Mobil08: I do not feel comfortable when I travel close to people I do not know.\n"
    p2 += "- Mobil09: Taking the bus helps making the city more comfortable and welcoming.\n"
    p2 += "- Mobil10: It is difficult to take the public transport when I travel with my children.\n"
    p2 += "- Mobil11: It is difficult to take the public transport when I carry bags or luggage.\n"
    p2 += "- Mobil12: It is very important to have a beautiful car.\n"
    p2 += "- Mobil13: With my car I can go wherever and whenever.\n"
    p2 += "- Mobil14: When I take the car I know I will be on time.\n"
    p2 += "- Mobil15: I do not like looking for a parking place.\n"
    p2 += "- Mobil16: I do not like changing the mean of transport when I am traveling.\n"
    p2 += "- Mobil17: If I use public transportation I have to cancel certain activities I would have done if I had taken the car.\n"
    p2 += "- Mobil18: CarPostal bus schedules are sometimes difficult to understand.\n"
    p2 += "- Mobil19: I know very well which bus/train I have to take to go where I want to.\n"
    p2 += "- Mobil20: I know by heart the schedules of the public transports I regularly use.\n"
    p2 += "- Mobil21: I can rely on my family to drive me if needed\n"
    p2 += "- Mobil22: When I am in a town I don’t know I feel strongly disoriented\n"
    p2 += "- Mobil23: I use the internet to check the schedules and the departure times of buses and trains.\n"
    p2 += "- Mobil24: I have always used public transports all my life\n"
    p2 += "- Mobil25: When I was young my parents took me to all my activities\n"
    p2 += "- Mobil26: I know some drivers of the public transports that I use\n"
    p2 += "- Mobil27: I think it is important to have the option to talk to the drivers of public transports.\n"
    
    # p3: Input information
    p3 = "Responses to the questions: " 
    p3 += "- Mobil01: " + str(row["Mobil01"]) + "\n"
    p3 += "- Mobil02: " + str(row["Mobil02"]) + "\n"
    p3 += "- Mobil03: " + str(row["Mobil03"]) + "\n"
    p3 += "- Mobil04: " + str(row["Mobil04"]) + "\n"
    p3 += "- Mobil05: " + str(row["Mobil05"]) + "\n"
    p3 += "- Mobil06: " + str(row["Mobil06"]) + "\n"
    p3 += "- Mobil07: " + str(row["Mobil07"]) + "\n"
    p3 += "- Mobil08: " + str(row["Mobil08"]) + "\n"
    p3 += "- Mobil09: " + str(row["Mobil09"]) + "\n"
    p3 += "- Mobil10: " + str(row["Mobil10"]) + "\n"
    p3 += "- Mobil11: " + str(row["Mobil11"]) + "\n"
    p3 += "- Mobil12: " + str(row["Mobil12"]) + "\n"
    p3 += "- Mobil13: " + str(row["Mobil13"]) + "\n"
    p3 += "- Mobil14: " + str(row["Mobil14"]) + "\n"
    p3 += "- Mobil15: " + str(row["Mobil15"]) + "\n"
    p3 += "- Mobil16: " + str(row["Mobil16"]) + "\n"
    p3 += "- Mobil17: " + str(row["Mobil17"]) + "\n"
    p3 += "- Mobil18: " + str(row["Mobil18"]) + "\n"
    p3 += "- Mobil19: " + str(row["Mobil19"]) + "\n"
    p3 += "- Mobil20: " + str(row["Mobil20"]) + "\n"
    p3 += "- Mobil21: " + str(row["Mobil21"]) + "\n"
    p3 += "- Mobil22: " + str(row["Mobil22"]) + "\n"
    p3 += "- Mobil23: " + str(row["Mobil23"]) + "\n"
    p3 += "- Mobil24: " + str(row["Mobil24"]) + "\n"
    p3 += "- Mobil25: " + str(row["Mobil25"]) + "\n"
    p3 += "- Mobil26: " + str(row["Mobil26"]) + "\n"
    p3 += "- Mobil27: " + str(row["Mobil27"]) + "\n"
    
    
    #p4: format output
    p4 = "Output 5 sentences that accurately describe the person's transportation behavior."
 
    

    
    final_prompt = prompt_template.invoke({
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
    })
    
    return final_prompt
    
# test prompt
if __name__ == "__main__":
    df = pd.read_csv("data/Optima/paperdata/behavior.csv")
    row = df.iloc[1]
    prompt_value = prompt(row)
    result = str(prompt_value)  # convert to string

    print(result)
    
    