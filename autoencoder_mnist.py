#%%

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

#%%
# 하이퍼파라미터
batch_size = 128
epochs = 20
learning_rate = 1e-3
folder_path = 'result'

# MNIST 데이터셋 로드
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# AutoEncoder 모델 정의
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 12),
            nn.GELU(),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.GELU(),
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), 1, 28, 28)
        return decoded

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
def train():
    model.train()
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    # 학습 종료 후 가중치 저장
    weight_path = os.path.join(folder_path, "weight")
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    torch.save(model.state_dict(), os.path.join(weight_path, 'autoencoder_weights_GELU.pth'))

# 테스트 및 결과 시각화
def test_and_plot():
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _, outputs = model(images)
            break  # 첫 배치만 시각화
    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    # result 폴더 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    fig, axes = plt.subplots(2, 8, figsize=(15, 3))
    for i in range(8):
        axes[0, i].imshow(images[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].imshow(outputs[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
    axes[0, 0].set_ylabel('Input')
    axes[1, 0].set_ylabel('Generated')
    plt.show()

    # 결과 저장 및 출력
    plt.savefig(os.path.join(folder_path, 'autoencoder_results_GELU.png'))

if __name__ == '__main__':
    # 학습 수행
    train()
    
    # 저장된 가중치 불러오기
    weight_path = os.path.join(folder_path, "weight", "autoencoder_weights_GELU.pth")
    model.load_state_dict(torch.load(weight_path))
    
    # 테스트 및 시각화
    test_and_plot()