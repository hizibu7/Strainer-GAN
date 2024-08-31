# 1,2,8 

import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms, models
from torchvision.utils import save_image
from torch.utils.data import Subset, ConcatDataset
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.models import inception_v3
from scipy import linalg
import torch.nn.functional as F

torch.use_deterministic_algorithms(False)

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

# 8 데이터 전체, 1과 2 데이터의 10%를 선택
num_one_samples = int(len(one_indices) * 0.1)
num_two_samples = int(len(two_indices) * 0.1)
num_eight_samples = len(eight_indices)

# 랜덤 샘플링
random.shuffle(one_indices)
random.shuffle(two_indices)
selected_one_indices = one_indices[:num_one_samples]
selected_two_indices = two_indices[:num_two_samples]
selected_eight_indices = eight_indices

# Subset 생성
one_dataset = Subset(MNIST_dataset, selected_one_indices)
two_dataset = Subset(MNIST_dataset, selected_two_indices)
eight_dataset = Subset(MNIST_dataset, selected_eight_indices)

# 세 데이터셋 결합
combined_dataset = ConcatDataset([one_dataset, two_dataset, eight_dataset])

# 데이터 개수 확인 및 배치 크기 조정
num_samples = len(combined_dataset)
print(f"Number of samples - 1: {num_one_samples}, 2: {num_two_samples}, 8: {num_eight_samples}, Total: {num_samples}")

# 배치 크기 설정 (최소 16, 최대 64)
batch_size = min(max(num_samples // 100, 16), 64)
print(f"Adjusted batch size: {batch_size}")

# 데이터 로더
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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_size, hidden_size1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size1),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size2),
            nn.Linear(hidden_size2, hidden_size3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_size3),
            nn.Linear(hidden_size3, img_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size, hidden_size3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size3, hidden_size2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, hidden_size1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size1, 1),
            nn.Sigmoid()
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


# ResNet 특징 추출기 정의
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.features(x).squeeze()

feature_extractor = FeatureExtractor().to(device)

# Z-score 계산 함수 수정
def compute_z_scores(dataset, feature_extractor):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    features = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feat = feature_extractor(images)
            features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    z_scores = np.abs((features - mean) / (std + 1e-7))
    max_z_scores = np.max(z_scores, axis=1)
    
    return max_z_scores


def get_inception_model():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    return inception_model.eval().to(device)

def get_activations(model, images, batch_size=50):
    n_batches = len(images) // batch_size
    act = []
    for i in range(n_batches):
        batch = images[i*batch_size:(i+1)*batch_size].to(device)
        if batch.size(1) == 1:
            batch = batch.repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
        with torch.no_grad():
            act_batch = model(batch)
            # 활성화 값 정규화
            act_batch = F.normalize(act_batch, p=2, dim=1)
            act.append(act_batch.cpu().numpy())
    act = np.concatenate(act, axis=0)
    return act

# FID 계산을 위한 데이터 로더 생성 (8만 포함)
eight_indices = [i for i, (_, label) in enumerate(MNIST_dataset) if label == 8]
eight_dataset = Subset(MNIST_dataset, eight_indices)
eight_loader = DataLoader(eight_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# FID 계산 함수 (이전에 제안한 수정사항 포함)
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    eps = 1e-6
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print("FID calculation produces singular product; adding epsilon to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid



# Clean dataset 생성 (1차 정제)
# 메인 코드에서 threshold 설정 및 데이터셋 정제
max_z_scores = compute_z_scores(combined_dataset, feature_extractor)

# 여기서 threshold를 직접 설정합니다
threshold = 4.0  # 예시 값, 필요에 따라 조정하세요

clean_indices = np.where(max_z_scores < threshold)[0]
clean_dataset = Subset(combined_dataset, clean_indices)

print(f"Original dataset size: {len(combined_dataset)}")
print(f"Clean dataset size: {len(clean_dataset)}")
print(f"Z-Score threshold: {threshold:.4f}")

# Z-score 분포 시각화 (선택사항)
plt.figure(figsize=(10, 6))
plt.hist(max_z_scores, bins=100, density=True, alpha=0.7)
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.title('Z-Score Distribution')
plt.xlabel('Max Z-Score')
plt.ylabel('Density')
plt.legend()
plt.show()


# 1차 정제된 데이터 로더
original_data_loader = DataLoader(dataset=clean_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)


# Inception 모델 로드
inception_model = get_inception_model()

# Training loop
for epoch in range(num_epoch):
    epoch_losses = []
    
    # 매 epoch마다 1차 정제 데이터셋으로 리셋
    current_data_loader = original_data_loader
    
    for i, (images, _) in enumerate(current_data_loader):
        current_batch_size = images.size(0)
    
        # 실제 레이블 생성 (현재 배치 크기에 맞춤)
        real_label = torch.full((current_batch_size, 1), 0.9, dtype=torch.float32).to(device)
        fake_label = torch.full((current_batch_size, 1), 0.1, dtype=torch.float32).to(device)

        # 실제 이미지 reshape
        real_images = images.reshape(current_batch_size, -1).to(device)

        # Train Discriminator
        d_optimizer.zero_grad()
        real_output = discriminator(real_images)
        real_loss = criterion(real_output, real_label)
        
        z = torch.randn(current_batch_size, noise_size).to(device)
        fake_images = generator(z)
        fake_output = discriminator(fake_images.detach())
        fake_loss = criterion(fake_output, fake_label)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()
        fake_output = discriminator(fake_images)
        g_loss = criterion(fake_output, real_label)
        g_loss.backward()
        g_optimizer.step()

        # 각 데이터 포인트의 loss 저장
        with torch.no_grad():
            individual_losses = criterion(real_output, real_label).view(-1)
            epoch_losses.extend(individual_losses.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}] Step [{i+1}/{len(current_data_loader)}] d_loss: {d_loss.item():.5f} g_loss: {g_loss.item():.5f}")
    
    # 3 epoch 이후부터 데이터 정제
    if epoch >= 3 and len(epoch_losses) > 0:
        # loss 임계값 계산 (상위 10%에 해당하는 loss 값)
        threshold = np.percentile(epoch_losses, 80)
        
        # 임계값보다 낮은 loss를 가진 데이터만 선택
        include_indices = [i for i, loss in enumerate(epoch_losses) if loss < threshold]
        
        # 새로운 데이터셋 생성
        new_dataset = Subset(clean_dataset, include_indices)
        
        # 새로운 데이터 로더 생성 (다음 epoch에서 사용)
        current_data_loader = DataLoader(dataset=new_dataset,
                                         batch_size=min(batch_size, len(new_dataset)),
                                         shuffle=True,
                                         drop_last=True)
        
        print(f"Epoch {epoch+1}: Prepared dataset for next epoch, excluded {len(clean_dataset) - len(new_dataset)} samples with losses above {threshold:.5f}")
    
    # 에폭마다 성능 출력
    with torch.no_grad():
        d_performance = discriminator(real_images).mean()
        g_performance = discriminator(fake_images).mean()
        print(f"Epoch {epoch+1}'s discriminator performance: {d_performance:.2f} generator performance: {g_performance:.2f}")

    # 10 에폭마다 FID 계산
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            num_samples = 1000  # FID 계산에 사용할 샘플 수
            real_images = []
            fake_images = []

            for _ in range(num_samples // batch_size):
                real_batch = next(iter(eight_loader))[0]  # 8만 포함된 데이터 로더 사용
                real_images.append(real_batch)
            
                z = torch.randn(batch_size, noise_size).to(device)
                fake_batch = generator(z).reshape(-1, 1, 28, 28)
                fake_images.append(fake_batch)

            real_images = torch.cat(real_images, dim=0)
            fake_images = torch.cat(fake_images, dim=0)

            # 이미지 정규화 확인 및 조정
            real_images = (real_images - 0.5) / 0.5  # [-1, 1] 범위로 조정
            fake_images = torch.clamp(fake_images, -1, 1)  # 생성된 이미지를 [-1, 1] 범위로 제한

            real_activations = get_activations(inception_model, real_images)
            fake_activations = get_activations(inception_model, fake_images)

            fid = calculate_fid(real_activations, fake_activations)
            print(f"Epoch {epoch+1}, FID (compared to 8s only): {fid}")

    # 가짜 이미지 저장
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(25, noise_size).to(device)
            fake_samples = generator(z)
            fake_samples = fake_samples.reshape(25, 1, 28, 28)
            save_image(fake_samples, os.path.join(dir_name, f'GAN_fake_samples_epoch{epoch + 1}.png'), nrow=5)

print("Training finished!")