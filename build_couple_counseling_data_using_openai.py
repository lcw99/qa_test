import os
import openai
import re, time

from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")

def openai_completion(prompt, temperature=0.6):
    err_count = 0
    while True:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=temperature,
                max_tokens=1355,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            break
        except:
            time.sleep(3)
            print("openai error")
            err_count += 1
            if err_count > 3:
                break
        
    result_text = response["choices"][0]["text"]
    result_text = result_text.replace("\n\n", "\n")
    result_text = result_text.strip()
    return result_text

data_folder = "couple_counseling_data"

# result_text = openai_completion("list 20 topics in detail for couples and boyfriend or girlfriend counseling in importance order.")
result_text = """1. Communication: Learning how to effectively communicate with each other, including active listening, expressing feelings, and resolving conflicts.
2. Trust: Building trust and understanding the importance of being honest and open with each other.
3. Intimacy: Discussing the importance of physical and emotional intimacy in a relationship.
4. Respect: Learning how to respect each other’s opinions, feelings, and boundaries.
5. Finances: Discussing financial goals and how to manage money together.
6. Family: Understanding the role of family in a relationship and how to navigate family dynamics.
7. Conflict Resolution: Learning how to resolve conflicts in a healthy and productive way.
8. Goals: Setting goals for the relationship and discussing how to achieve them.
9. Boundaries: Establishing and respecting boundaries in the relationship.9
10. Self-Care: Discussing the importance of self-care and how to prioritize it in the relationship.
11. Stress Management: Learning how to manage stress and how to support each other during difficult times.
12. Compromise: Understanding the importance of compromise and how to negotiate differences.
13. Values: Discussing values and how they impact the relationship.
14. Priorities: Identifying and prioritizing goals and values in the relationship.
15. Self-Awareness: Learning how to be self-aware and understanding the impact of one’s actions on the relationship.
16. Listening: Developing active listening skills and understanding the importance of listening to each other.
17. Forgiveness: Learning how to forgive and move forward after a mistake or conflict.
18. Support: Understanding the importance of providing emotional support to each other.
19. Problem-Solving: Learning how to identify and solve problems in the relationship.
20. Self-Esteem: Discussing the importance of self-esteem and how to build it in the relationship."""
main_topics = result_text.split("\n")
main_topics = main_topics[6:]
for mt in main_topics:
    mt = re.sub("\d+.\s", "", mt).strip()
    if mt == "":
        continue
    mt_fn = mt.replace(": ", "-")
    mt_fn = mt_fn.replace(" ", "_")
    mt_fn = mt_fn + "txt"
    prompt = "list 20 sub-topics about the topics below that may arise in couples and boyfriend or girlfriend relationships.\nTopic: "
    prompt = prompt + mt
    result_text = openai_completion(prompt)
    sub_topics = result_text.split("\n")
    for st in sub_topics:
        st = re.sub("\d+.\s", "", st).strip()
        if st == "":
            continue
        st_fn = st.replace(": ", "-")
        st_fn = st_fn.replace(" ", "_")
        
        for i in range(2):
            timestamp = int(round(datetime.now().timestamp()))
            st_fn_ts = f"{st_fn}_{timestamp}.txt"
            print(st_fn_ts)
            prompt = "Write articles about the topics and sub-topics below that may arise in couples and boyfriend or girlfriend relationships. Be sure to include a title and subtitle.\nTopic:"
            prompt = prompt + mt + "\nSub-topic: " + st        
            result_text = openai_completion(prompt, temperature=0.9)
            
            out = open(f"{data_folder}/{st_fn_ts}", "w")
            out.write(result_text)
            out.close()
    