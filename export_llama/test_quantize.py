import torch
import torch.nn as nn
import torch.optim as optim
from quantize import quantize
# Define your simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

device="cpu"
input_size = 1000
hidden_size = 1000
output_size = 10
model = SimpleModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
batch_size = 1024
input_data = torch.randn(batch_size, input_size).to(device)
target_labels = torch.randn(batch_size, output_size).to(device)
for epoch in range(num_epochs):
    output = model(input_data)
    loss = criterion(output, target_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

model.eval()
# test_data = torch.randn(1, input_size).to(device)
test_data = input_data
with torch.no_grad():
    output = model(input_data)
    loss = criterion(output, target_labels)
    print("No quantization, Prediction:", loss)
    
torch.onnx.export(model,test_data,"test_no_quant.onnx")
quantize_cfg = {
    "fc1":{
        "type":"W8DX",
    },"fc2":{
        "type":"W8DX",
    },"fc3":{
        "type":"W8DX",
    }
}
quantize(model,cfg=quantize_cfg)

with torch.no_grad():
    output = model(input_data)
    loss = criterion(output, target_labels)
    print("Int8 quantization, Prediction:", loss)
    
test_data = torch.randn(1, input_size).to(device)
torch.onnx.export(model,test_data,"test_quant.onnx",opset_version=13)
