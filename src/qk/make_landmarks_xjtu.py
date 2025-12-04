# make_landmarks_xjtu.py
import os, torch
from main_XJTU_qk import load_xjtu_tensors, pick_landmarks

os.makedirs('data', exist_ok=True)
xt_tr, y_tr, gid_tr, *rest = load_xjtu_tensors()
x_tr = xt_tr[:, :-1]
lm = pick_landmarks(x_tr, M=256)
torch.save(lm, 'data/qk_landmarks_xjtu.pt')
print("Saved landmarks -> data/qk_landmarks_xjtu.pt")
