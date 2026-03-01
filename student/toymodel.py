import torch
import torch.nn as nn
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

'''device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToyModel(in_features=8, out_features=4).to(device)
print("Parameter dtypes (before autocast):")
for name, p in model.named_parameters():
    print(f"  {name}: {p.dtype}")

x = torch.randn(2, 8, device=device)  # FP32 input

print("\nInside autocast:")
with torch.autocast(device_type=device, dtype=torch.float16):
    # forward pass step by step
    out_fc1 = model.relu(model.fc1(x))
    print(f"  fc1 output dtype:    {out_fc1.dtype}")

    out_ln = model.ln(out_fc1)
    print(f"  ln  output dtype:    {out_ln.dtype}")

    logits = model.fc2(out_ln)
    print(f"  logits dtype:        {logits.dtype}")

    loss = logits.sum()
    print(f"  loss dtype:          {loss.dtype}")

print("\nParameter dtypes (after autocast, before backward):")
for name, p in model.named_parameters():
    print(f"  {name}: {p.dtype}")

loss.backward()

print("\nGradient dtypes (after backward):")
for name, p in model.named_parameters():
    print(f"  {name}.grad: {p.grad.dtype}")'''