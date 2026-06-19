"""
Fine-tune BreizhCrops TempCNN on Irish S2 time series.
Loads pretrained weights, replaces head, trains on Irish parcels.
"""
import torch, torch.nn as nn, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from breizhcrops import TempCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

# Load Irish S2 time series
with open('/workspaces/crop-trajectory/data/irish_s2_timeseries.json') as f:
    data = json.load(f)

print(f"Loaded {len(data)} Irish parcels")
from collections import Counter
print("Classes:", Counter(d['label'] for d in data))

# Build tensors
X_list=[]; y_list=[]
labels=[]
for d in data:
    ts = np.array(d['timeseries'], dtype=np.float32)  # [32, 13]
    if ts.shape[0]<10: continue
    X_list.append(ts)
    labels.append(d['label'])

X = torch.tensor(np.stack(X_list)).float()  # [N, 32, 13]
le = LabelEncoder()
y = torch.tensor(le.fit_transform(labels)).long()
print(f"X:{X.shape} classes:{list(le.classes_)}")

# Load pretrained TempCNN (trained on 45-step sequences)
# Irish data is 32 steps — create model with sequencelength=32
n_classes = len(le.classes_)
model = TempCNN(input_dim=13, num_classes=5, sequencelength=32)

# Load pretrained weights with strict=False (head size mismatch)
pretrained = torch.load(
    '/workspaces/crop-trajectory/models/tempcnn_breizhcrops.pth',
    map_location='cpu')
# Only load conv layers, not the final head
conv_weights = {k:v for k,v in pretrained.items()
                if not k.startswith('logsoftmax')}
model.load_state_dict(conv_weights, strict=False)
print("Pretrained conv weights loaded")

# Replace head for Irish classes
model.logsoftmax = nn.Sequential(
    nn.Linear(4*128, n_classes),
    nn.LogSoftmax(dim=-1)
)

# Train/test split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(sss.split(X, y))
X_tr,y_tr = X[tr],y[tr]
X_te,y_te = X[te],y[te]

# Fine-tune: first freeze conv, train head only
for p in model.parameters(): p.requires_grad=False
for p in model.logsoftmax.parameters(): p.requires_grad=True

optimizer = torch.optim.Adam(model.logsoftmax.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

print("\nPhase 1: Train head only (frozen conv)...")
for epoch in range(20):
    model.train(); p=torch.randperm(len(X_tr))
    ls=0; correct=0
    for i in range(0,len(X_tr),32):
        b=p[i:i+32]; xb=X_tr[b]; yb=y_tr[b]
        optimizer.zero_grad()
        out=model(xb); loss=criterion(out,yb)
        loss.backward(); optimizer.step()
        ls+=loss.item(); correct+=(out.argmax(1)==yb).sum().item()
    if (epoch+1)%5==0:
        print(f"  Epoch {epoch+1}: train={correct/len(X_tr)*100:.1f}%")

# Phase 2: Unfreeze all, fine-tune end-to-end
for p in model.parameters(): p.requires_grad=True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("\nPhase 2: End-to-end fine-tuning...")
for epoch in range(30):
    model.train(); p=torch.randperm(len(X_tr))
    ls=0; correct=0
    for i in range(0,len(X_tr),32):
        b=p[i:i+32]; xb=X_tr[b]; yb=y_tr[b]
        optimizer.zero_grad()
        out=model(xb); loss=criterion(out,yb)
        loss.backward(); optimizer.step()
        ls+=loss.item(); correct+=(out.argmax(1)==yb).sum().item()
    if (epoch+1)%5==0:
        model.eval()
        with torch.no_grad():
            out=model(X_te); preds=out.argmax(1)
            acc=(preds==y_te).float().mean().item()*100
        print(f"  Epoch {epoch+1}: train={correct/len(X_tr)*100:.1f}% test={acc:.1f}%")
        model.train()

model.eval()
with torch.no_grad():
    out=model(X_te); preds=out.argmax(1)
    acc=(preds==y_te).float().mean().item()*100
print(f"\nFinal accuracy: {round(acc,1)}%")
print(classification_report(y_te.numpy(), preds.numpy(), target_names=le.classes_))
print(f"v15 CatBoost baseline: 63.5%")
