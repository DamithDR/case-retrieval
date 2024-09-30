import json

import pandas as pd

with open('data/files/muser/muser_cases_pool.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# concat the facts findings and court opinion
rows = [{'id': key, 'text': '\n'.join(value['content']['本院查明']) + '\n'.join(value['content']['本院认为'])} for
        key, value in data.items()]
df = pd.DataFrame(rows)

print(df)
