import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from text_ctc_utils import *
from torch.utils.data import DataLoader, TensorDataset
from data_loader import HandPoseDataset
from ctc_decoder import Decoder

data_dir = "/home/ksw38/MachineTranslation/mediapipe_res_chicago/"
hand_detected_label = "/home/ksw38/MachineTranslation/fingerspelling-posenet/sign_hand_detection_wild.csv"
labels_csv = "/home/ksw38/MachineTranslation/data/ChicagoFSWild/ChicagoFSWild.csv"

def enable_mc_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def mc_dropout_inference(model, inputs, T=20):
    model.eval()
    enable_mc_dropout(model)  # Activate dropout

    logits_list = []
    with torch.no_grad():
        for _ in range(T):
            _, logits = model(inputs)  # [T, B, C]
            logits_list.append(logits.unsqueeze(0))  # [1, T, B, C]

    all_logits = torch.cat(logits_list, dim=0)  # [T, T_seq, B, C]
    return all_logits  # You can now compute mean & std over T
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

beam_size  = 5
lm_beta = 0.4
ins_gamma = 1.2
chars = "$' &.@acbedgfihkjmlonqpsrutwvyxz"
vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)
target_enc_df = convert_text_for_ctc(labels_csv,vocab_map,True)

model_pth = "best_model_66.3.pt"
model = TransformerModel(output_dim=len(char_list), d_input = 42 ,d_model=256, nhead=8, num_layers=3, dropout=0.1).to(device)
model.load_state_dict(torch.load(model_pth))
model.eval()

dataset_test = HandPoseDataset(data_dir, labels_csv , hand_detected_label, target_enc_df , "test" , augmentations =False )
testdataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)
decoder_dec = Decoder(char_list, blank_index=0)

for i, (poses, labels) in enumerate(testdataloader):
    X_batch = poses.to(device)

    # MC Dropout Inference
    all_logits = mc_dropout_inference(model, X_batch, T=20)
    mean_logits = all_logits.mean(dim=0)  # [T, B, C]

    # Continue as usual
    log_probs_enc = F.log_softmax(mean_logits, dim=-1)
    log_probs_enc = log_probs_enc.permute(1, 0, 2)  # [B, T, C] → [T, B, C] if needed

    # Decode, compute loss, etc.
    current_preds = decoder_dec.beam_decode_trans(
        log_probs_enc[:, 0, :].detach().cpu(), 
        beam_size, 
        model, 
        poses, 
        beta=lm_beta, 
        gamma=ins_gamma
    )
    
    print(f"Bayesian Prediction: {''.join(current_preds)}")


