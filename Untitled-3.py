# 1(10) / 2(10) / 8(80)

import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset, Dataset
import random

# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Now using {device} device")

# Create a directory for saving samples
dir_name = "GAN_results"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

# MNIST dataset setting
MNIST_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transform,
                                           download=True)

# 숫자 1, 2, 8인 데이터의 인덱스 선택
one_indices = [i for i, (_, label) in enumerate(MNIST_dataset) if label == 1]
two_indices = [i for i, (_, label) in enumerate(MNIST_dataset) if label == 2]
eight_indices = [i for i, (_, label) in enumerate(MNIST_dataset) if label == 8]

# 전체 데이터셋의 10%를 1로, 10%를 2로, 80%를 8로 구성
total_samples = min(len(one_indices), len(two_indices)) * 10  # 가장 적은 클래스를 기준으로 설정
num_one_samples = int(total_samples * 0.1)
num_two_samples = int(total_samples * 0.1)
num_eight_samples = total_samples - num_one_samples - num_two_samples

# 랜덤 샘플링
random.shuffle(one_indices)
random.shuffle(two_indices)
random.shuffle(eight_indices)
selected_one_indices = one_indices[:num_one_samples]
selected_two_indices = two_indices[:num_two_samples]
selected_eight_indices = eight_indices[:num_eight_samples]

# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# 커스텀 데이터셋 생성
one_dataset = CustomDataset(MNIST_dataset, selected_one_indices)
two_dataset = CustomDataset(MNIST_dataset, selected_two_indices)
eight_dataset = CustomDataset(MNIST_dataset, selected_eight_indices)

# 세 데이터셋 결합
combined_dataset = ConcatDataset([one_dataset, two_dataset, eight_dataset])

# 데이터 개수 확인 및 배치 크기 조정
num_samples = len(combined_dataset)
print(f"Number of samples - 1: {num_one_samples}, 2: {num_two_samples}, 8: {num_eight_samples}, Total: {num_samples}")

# 배치 크기 설정 (최소 16, 최대 64)
batch_size = min(max(num_samples // 100, 16), 64)
print(f"Adjusted batch size: {batch_size}")

# 데이터 로더 (셔플 옵션 True로 설정)
data_loader = torch.utils.data.DataLoader(dataset=combined_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# Hyper-parameters
num_epoch = 300
learning_rate = 0.0002
img_size = 28 * 28
noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, hidden_size3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size3, hidden_size2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size2, hidden_size1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, img_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Initialize generator/Discriminator
discriminator = Discriminator().to(device)
generator = Generator().to(device)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epoch):
    for i, (images, _) in enumerate(data_loader):
        current_batch_size = images.size(0)

        # 실제 레이블 생성 (현재 배치 크기에 맞춤)
        real_label = torch.full((current_batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((current_batch_size, 1), 0, dtype=torch.float32).to(device)

        # 실제 이미지 reshape
        real_images = images.reshape(current_batch_size, -1).to(device)

        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(current_batch_size, noise_size).to(device)
        fake_images = generator(z)
        g_loss = criterion(discriminator(fake_images), real_label)
        g_loss.backward()
        g_optimizer.step()

        # Train Discriminator
        d_optimizer.zero_grad()
        real_loss = criterion(discriminator(real_images), real_label)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}] Step [{i+1}/{len(data_loader)}] d_loss: {d_loss.item():.5f} g_loss: {g_loss.item():.5f}")

    # 에폭마다 성능 출력
    with torch.no_grad():
        d_performance = discriminator(real_images).mean()
        g_performance = discriminator(fake_images).mean()
        print(f"Epoch {epoch+1}'s discriminator performance: {d_performance:.2f} generator performance: {g_performance:.2f}")

    # 가짜 이미지 저장
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            samples = fake_images[:25].reshape(25, 1, 28, 28)  # 25개의 샘플만 저장
            save_image(samples, os.path.join(dir_name, f'GAN_fake_samples_epoch{epoch + 1}.png'), nrow=5)

print("Training finished!")