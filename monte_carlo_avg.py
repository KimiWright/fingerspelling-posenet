import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from text_ctc_utils import *
from torch.utils.data import DataLoader, TensorDataset
from data_loader import HandPoseDataset
from ctc_decoder import Decoder
from utils import *
import csv

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

beta_values = [1.9, 2.0, 2.1, 2.2, 2.3]
gamma_values = [0.6, 0.8]
T_values = [1, 2, 100, 200, 300]
params = [(beta, gamma, T) for beta in beta_values for gamma in gamma_values for T in T_values]

best_lev_acc = float('-inf')
best_lev_acc_beam = float('-inf')
best_params = None

for lm_beta, ins_gamma, T in params:
    print("LM Beta: ", lm_beta, " Ins Gamma: ", ins_gamma, " T: ", T)

    preds = []
    gt_labels = []
    preds_encoder = []
    for i, (poses, labels) in enumerate(testdataloader):
        X_batch = poses.to(device)

        # MC Dropout Inference
        all_logits = mc_dropout_inference(model, X_batch, T=T)
        # mean_logits = all_logits.mean(dim=0)  # [T, B, C]
        

        # # Continue as usual
        # log_probs_enc = F.log_softmax(mean_logits, dim=-1)
        # log_probs_enc = log_probs_enc.permute(1, 0, 2)  # [B, T, C] → [T, B, C] if needed
        probs = F.softmax(all_logits, dim=-1)
        avg_probs = probs.mean(dim=0)  # [B, T, C]
        log_probs_enc = torch.log(avg_probs + 1e-10)  # Add small value to avoid log(0)
        log_probs_enc = log_probs_enc.permute(1, 0, 2)  # [B, T, C] → [T, B, C] if needed

        current_preds_enc = ""
        preds_encoder.append(current_preds_enc)

        cls_token, _ = model(poses)

        pred_size = (torch.atan2(torch.tensor([cls_token[0,0].detach().cpu()]),torch.tensor([cls_token[0,1].detach().cpu()]))/(2 * torch.pi) +0.5) * 30
        pred_size = torch.round(pred_size)

        # Decode, compute loss, etc.
        current_preds = decoder_dec.beam_decode_trans(
            log_probs_enc[:, 0, :].detach().cpu(), 
            beam_size, 
            model, 
            poses, 
            beta=lm_beta, 
            gamma=ins_gamma
        )
        current_preds = ''.join(current_preds)
        
        preds.append(current_preds)
        # print("DecLM : ", current_preds, " EN : " , current_preds_enc, " GT : "  , ''.join(invert_to_chars(labels[:,1:-1],inv_vocab_map)), "   ", pred_size) 
        gt_labels.append(''.join(invert_to_chars(labels[:,1:-1],inv_vocab_map)))

    lev_acc = compute_acc(preds_encoder, gt_labels)
    lev_acc_beam = compute_acc(preds, gt_labels)
    if lev_acc > best_lev_acc:
        best_lev_acc = lev_acc
        best_params = (lm_beta, ins_gamma, T)
    if lev_acc_beam > best_lev_acc_beam:
        best_lev_acc_beam = lev_acc_beam


    print('Letter Acc: {:.4f} - Best Acc {:.4f}'.format(lev_acc, lev_acc_beam))
    with open('avg_results4.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([lm_beta, ins_gamma, T, lev_acc, lev_acc_beam])
    

