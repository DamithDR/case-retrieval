from data.DataClass import DataClass
from data.IL_PCR import IL_PCR

if __name__ == '__main__':
    dataset = IL_PCR(32000)
    dataset.vectorise_candidates()