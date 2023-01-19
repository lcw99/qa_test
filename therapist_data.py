# Import pandas
import pandas as pd

# Read CSV file into DataFrame
df = pd.read_csv('/home/chang/AI/counsel-chat/data/20200325_counsel_chat.csv')
df = df.reset_index()  # make sure indexes pair with number of rows

for index, row in df.iterrows():
    out = open(f"data_therapist_files/therapist_qa_{index}.txt", "w")
    # q = "Customer: " + row['questionText']
    a = row['answerText']
    # out.write(f"{q}\n")
    out.write(f"{a}\n\n")
    out.close()