import json

import pandas as pd



file_name = 'data/files/irled/candidates/prior_case_0001.txt'
with open(file_name, 'r', encoding='utf-8') as file:
    text = file.read()
    print(text)
# for filename in os.listdir(path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(path, filename)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             text = file.read()
#         ids.append(os.path.splitext(filename)[0])
#         data.append(text)
# return ids, text


