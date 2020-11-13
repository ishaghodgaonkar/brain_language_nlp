import numpy as np
import torch
import time as tm
from pytorch_transformers import XLNetConfig, XLNetModel, XLNetTokenizer
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel
import time as tm


configuration = XLNetConfig(mem_len=1600)
model = XLNetModel(configuration)
configuration = model.config
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
model.eval()
print(configuration)


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()
config = model.config
print(config)