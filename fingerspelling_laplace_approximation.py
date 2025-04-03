from laplace import Laplace
from model import *
from text_ctc_utils import *

chars = "$' &.@acbedgfihkjmlonqpsrutwvyxz"
vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pth = "best_model_66.3.pt"
model = TransformerModel(output_dim=len(char_list), d_input = 42 ,d_model=256, nhead=8, num_layers=3, dropout=0.1).to(device)

la = Laplace(model, 'regression', subset_of_weights='last_layer', hessian_structure='full')