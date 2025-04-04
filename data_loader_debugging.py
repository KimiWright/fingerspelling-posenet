import pandas as pd
from data_loader import HandPoseDataset
from text_ctc_utils import *
from torch.utils.data import DataLoader

data_dir = "/home/ksw38/MachineTranslation/mediapipe_res_chicago/"
hand_detected_label = "/home/ksw38/MachineTranslation/fingerspelling-posenet/sign_hand_detection_wild.csv" # Also try first if this doesn't work
labels_csv = "/home/ksw38/MachineTranslation/data/ChicagoFSWild/ChicagoFSWild.csv"

chars = "$' &.@acbedgfihkjmlonqpsrutwvyxz"
vocab_map, inv_vocab_map, char_list = get_autoreg_vocab(chars)
target_enc_df = convert_text_for_ctc(labels_csv,vocab_map,True)

dataset_test = HandPoseDataset(data_dir, labels_csv , hand_detected_label, target_enc_df , "test" , augmentations =False )
testdataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)

for i, (poses, labels) in enumerate(testdataloader):
    print(labels)