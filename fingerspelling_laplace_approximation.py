### Broken, laplace redux may not be a functional library of CTCLoss ###
from laplace import Laplace
from model import *
from text_ctc_utils import *
from torch.utils.data import DataLoader, TensorDataset
from data_loader import HandPoseDataset
from ctc_decoder import Decoder

data_dir = "/home/ksw38/MachineTranslation/mediapipe_res_chicago/"
hand_detected_label = "/home/ksw38/MachineTranslation/fingerspelling-posenet/sign_hand_detection_wild.csv"
labels_csv = "/home/ksw38/MachineTranslation/data/ChicagoFSWild/ChicagoFSWild.csv"

class LogitsOnlyModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        _, logits = self.base_model(x)  # discard cls_token
        return logits.permute(1, 0, 2)

beam_size  = 5
lm_beta = 0.4
ins_gamma = 1.2
chars = "$' &.@acbedgfihkjmlonqpsrutwvyxz"
vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)
target_enc_df = convert_text_for_ctc(labels_csv,vocab_map,True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model_pth = "best_model_66.3.pt"
model = TransformerModel(output_dim=len(char_list), d_input = 42 ,d_model=256, nhead=8, num_layers=3, dropout=0.1).to(device)
model.load_state_dict(torch.load(model_pth)) # Comment out for cpu debugging
model.eval()
wrapped_model = LogitsOnlyModel(model).to(device)

dataset_test = HandPoseDataset(data_dir, labels_csv , hand_detected_label, target_enc_df , "test" , augmentations =False )
testdataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)
decoder_dec = Decoder(char_list, blank_index=0) # This is the line that produces the lsit of characters

# for poses, labels in testdataloader:
#     poses = poses.to(device)
#     labels = labels.to(device)
#     outputs = model(poses)[1]
#     print(outputs.shape)
#     print(labels.shape)
#     print(outputs)
#     print(labels)
#     break

la = Laplace(wrapped_model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
print("Laplace Approximation initialized.")
# la.fit(testdataloader)
# print("Laplace Approximation fit complete.")

# X_batch = []
# for i, (x, _) in enumerate(testdataloader):
#     X_batch.append(x)
#     if i >= 10:  # just a few batches are enough for a Hessian approx
#         break
# X_batch = torch.cat(X_batch).to(device)

# dummy_y = torch.randn(X_batch.size(0), model.output_dim).to(device)

# la.fit(X_batch, dummy_y)

X_batch, _ = next(iter(testdataloader))
X_batch = X_batch.to(device)

# create dummy targets for regression
dummy_y = torch.randn(X_batch.size(0), len(char_list)).to(device)

# la.fit(X=X_batch, y =dummy_y)

# Wrap tensors into dataset
dataset = TensorDataset(X_batch, dummy_y)

# Create a DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Fit Laplace
la.fit(loader)

# # Optimize prior precision (lambda) using Empirical Bayes
# la.optimize_prior_precision()

# # Bayesian inference during evaluation
# for i, (poses, labels) in enumerate(testdataloader):
#     poses = poses.to(device)
    
#     # Bayesian forward pass
#     logits = la(poses)  
    
#     # Convert to probability space
#     log_probs_enc = F.log_softmax(logits, dim=-1).permute(1,0,2)
    
#     # Decode predictions using beam search
#     current_preds = decoder_dec.beam_decode_trans(
#         log_probs_enc[:, 0, :].detach().cpu(), 
#         beam_size, 
#         model, 
#         poses, 
#         beta=lm_beta, 
#         gamma=ins_gamma
#     )
    
#     print(f"Bayesian Prediction: {''.join(current_preds)}")