# loss만 + concate + fast + 10%

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.models import resnet18
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy import linalg
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture
from PIL import Image
from torchvision.models import resnet50
from scipy import linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from torch.utils.data import Dataset, DataLoader

# 코드 실행결과의 동일성을 위해 무작위 시드를 설정합니다
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 만일 새로운 결과를 원한다면 주석을 없애면 됩니다
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # 결과 재현을 위해 필요합니다


# 데이터셋의 경로
dataroot = "celeba"

# dataloader에서 사용할 쓰레드 수
workers = 2

# 배치 크기
batch_size = 128

# 이미지의 크기입니다. 모든 이미지를 변환하여 64로 크기가 통일됩니다.
image_size = 64

# 이미지의 채널 수로, RGB 이미지이기 때문에 3으로 설정합니다.
nc = 3

# 잠재공간 벡터의 크기 (예. 생성자의 입력값 크기)
nz = 100

# 생성자를 통과하는 특징 데이터들의 채널 크기
ngf = 64

# 구분자를 통과하는 특징 데이터들의 채널 크기
ndf = 64

# 학습할 에폭 수
num_epochs = 10

# 옵티마이저의 학습률
lr = 0.0002

# Adam 옵티마이저의 beta1 하이퍼파라미터
beta1 = 0.5

# 사용가능한 gpu 번호. CPU를 사용해야 하는경우 0으로 설정하세요
ngpu = 1



celeba_dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


# Anime Face Dataset을 로드하는 함수
def load_anime_faces(root_dir):
    images = []
    for file in os.listdir(root_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root_dir, file)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except:
                print(f"Error loading image: {img_path}")
    return images

# Anime Face Dataset 로드
anime_root = "anime"  # Anime Face Dataset의 경로를 지정해주세요
anime_images = load_anime_faces(anime_root)

# Anime 이미지를 텐서로 변환
anime_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

anime_tensors = [anime_transform(img) for img in anime_images]

# CelebA와 Anime Face Dataset 결합
class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, celeba_dataset, anime_tensors):
        self.celeba_dataset = celeba_dataset
        self.anime_tensors = anime_tensors

    def __len__(self):
        return len(self.celeba_dataset) + len(self.anime_tensors)

    def __getitem__(self, index):
        if index < len(self.celeba_dataset):
            return self.celeba_dataset[index]
        else:
            anime_index = index - len(self.celeba_dataset)
            return self.anime_tensors[anime_index], 1  # 1은 Anime Face의 레이블

combined_dataset = CombinedDataset(celeba_dataset, anime_tensors)

# DataLoader 정의
dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

print(f"CelebA dataset size: {len(celeba_dataset)}")
print(f"Anime Face dataset size: {len(anime_tensors)}")
print(f"Total number of samples: {len(combined_dataset)}")





# GPU 사용여부를 결정해 줍니다
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 학습 데이터들 중 몇가지 이미지들을 화면에 띄워봅시다
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()




# ``netG`` 와 ``netD`` 에 적용시킬 커스텀 가중치 초기화 함수
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




# 생성자 코드

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력데이터 Z가 가장 처음 통과하는 전치 합성곱 계층입니다.
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 위의 계층을 통과한 데이터의 크기. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 위의 계층을 통과한 데이터의 크기. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)





# 생성자를 만듭니다
netG = Generator(ngpu).to(device)

# 필요한 경우 multi-GPU를 설정 해주세요
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해
# ``weight_init`` 함수를 적용시킵니다
netG.apply(weights_init)





# 구분자 코드

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 입력 데이터의 크기는 ``(nc) x 64 x 64`` 입니다
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 위의 계층을 통과한 데이터의 크기. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)




# 구분자를 만듭니다
netD = Discriminator(ngpu).to(device)

# 필요한 경우 multi-GPU를 설정 해주세요
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 모든 가중치의 평균을 0( ``mean=0`` ), 분산을 0.02( ``stdev=0.02`` )로 초기화하기 위해
# ``weight_init`` 함수를 적용시킵니다
netD.apply(weights_init)




# ``BCELoss`` 함수의 인스턴스를 초기화합니다
criterion = nn.BCELoss()

# 생성자의 학습상태를 확인할 잠재 공간 벡터를 생성합니다
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 학습에 사용되는 참/거짓의 라벨을 정합니다
real_label = 1.
fake_label = 0.

# G와 D에서 사용할 Adam옵티마이저를 생성합니다
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



def find_elbow_threshold(z_scores, bins=100):
    # z-score 정렬 및 빈도 계산
    hist, bin_edges = np.histogram(z_scores, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 가장 density가 높은 지점 찾기
    peak_index = np.argmax(hist)
    peak_z_score = bin_centers[peak_index]

    # density가 0.01이 되는 오른쪽 지점 찾기
    right_side_hist = hist[peak_index:]
    right_side_bins = bin_centers[peak_index:]
    target_index = np.argmin(np.abs(right_side_hist - 0.01))
    target_z_score = right_side_bins[target_index]

    # 중간 지점을 threshold로 설정
    threshold = (peak_z_score + target_z_score) / 2

    return threshold, bin_centers, hist

def visualize_z_scores(z_scores, threshold, bin_centers, hist):
    plt.figure(figsize=(12, 6))
    
    # 원본 히스토그램
    plt.hist(z_scores, bins=100, density=True, alpha=0.7, color='skyblue', label='Distribution')
    
    if bin_centers is not None and hist is not None:
        # 스무딩된 곡선
        plt.plot(bin_centers, hist, color='navy', label='Density')
    
    # threshold 선
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
    
    plt.xlabel('Z-Score')
    plt.ylabel('Density')
    plt.title('Distribution of Z-Scores with Threshold')
    plt.legend()
    plt.show()


def detect_outliers(dataset, feature_extractor, user_threshold=None):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            img = batch[0].to(device)
            feat = feature_extractor(img)
            features.append(feat.cpu())
    
    features = torch.cat(features, dim=0)
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    
    z_scores = torch.abs((features - mean) / std)
    max_z_scores = z_scores.max(dim=1)[0].numpy()
    
    if user_threshold is None:
        threshold, bin_centers, hist = find_elbow_threshold(max_z_scores)
        print(f"Calculated threshold: {threshold}")
    else:
        threshold = user_threshold
        print(f"Using user-specified threshold: {threshold}")
        bin_centers, hist = None, None
    
    # z-score 분포 시각화
    visualize_z_scores(max_z_scores, threshold, bin_centers, hist)
    
    is_inlier = max_z_scores < threshold
    return is_inlier



def refine_dataset_by_loss(dataset, discriminator, device, loss_ratio=0.2):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    losses = []
    
    discriminator.eval()
    criterion = nn.BCELoss(reduction='none')
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = discriminator(images)
            loss = criterion(outputs, torch.ones_like(outputs)).mean(dim=1)
            losses.extend(loss.cpu().numpy())
    
    losses = np.array(losses)
    
    # 사용자가 지정한 비율에 따라 임계값 계산
    threshold = np.percentile(losses, (1 - loss_ratio) * 100)
    
    # 임계값보더 작은 loss를 가진 데이터만 선택
    clean_indices = np.where(losses < threshold)[0]
    
    # 만약 clean_indices가 비어있다면, 전체 데이터셋의 하위 50%를 선택
    if len(clean_indices) == 0:
        clean_indices = np.argsort(losses)[:max(len(dataset)//2, 1)]  # 최소 1개 이상의 샘플 유지
    
    clean_dataset = torch.utils.data.Subset(dataset, clean_indices)
    
    return clean_dataset, threshold




def get_activations(images, model, batch_size=50, device='cpu'):
    model.eval()
    n_batches = len(images) // batch_size
    n_used_imgs = n_batches * batch_size
    
    # 첫 번째 배치를 실행하여 실제 출력 차원을 확인합니다
    with torch.no_grad():
        first_batch = images[:batch_size].to(device)
        first_batch = resize_images(first_batch)
        first_pred = model(first_batch)
        if isinstance(first_pred, tuple):
            first_pred = first_pred[0]
        dims = first_pred.shape[1]
    
    pred_arr = np.empty((n_used_imgs, dims))
    
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end].to(device)
        batch = resize_images(batch)
        with torch.no_grad():
            pred = model(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
        pred_arr[start:end] = pred.cpu().numpy().reshape(batch_size, -1)
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):  # 1e-3에서 1e-2로 변경
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid(real_images, fake_images, batch_size=50, device='cpu'):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()
    
    # 이미지 크기 조정
    real_images = resize_images(real_images)
    fake_images = resize_images(fake_images)
    
    real_activations = get_activations(real_images, inception_model, batch_size, device=device)
    fake_activations = get_activations(fake_images, inception_model, batch_size, device=device)
    
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
    
    # 공분산 행렬에 작은 값을 더합니다
    sigma_real += np.eye(sigma_real.shape[0]) * 1e-6
    sigma_fake += np.eye(sigma_fake.shape[0]) * 1e-6
    
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_value

def resize_images(images):
    return F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)




# ResNet50을 특징 추출기로 사용
def load_feature_extractor():
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()  # 마지막 fully connected 층 제거
    return model.to(device)

def extract_features(images, feature_extractor):
    features = []
    feature_extractor.eval()
    with torch.no_grad():
        for image in images:
            feat = feature_extractor(image.unsqueeze(0).to(device))
            features.append(feat.cpu().numpy())
    return np.array(features)

def calculate_feature_distance(features1, features2):
    return np.linalg.norm(np.mean(features1, axis=0) - np.mean(features2, axis=0))

def calculate_wasserstein_distance(features1, features2):
    # Check and adjust the shape of input features
    if features1.ndim > 2:
        features1 = features1.reshape(features1.shape[0], -1)
    if features2.ndim > 2:
        features2 = features2.reshape(features2.shape[0], -1)
    
    # Reduce dimensionality to speed up Wasserstein distance calculation
    pca = PCA(n_components=min(50, features1.shape[1], features2.shape[1]))
    features1_pca = pca.fit_transform(features1)
    features2_pca = pca.transform(features2)
    
    # Calculate Wasserstein distance for each dimension
    distances = [wasserstein_distance(features1_pca[:, i], features2_pca[:, i]) for i in range(features1_pca.shape[1])]
    
    # Return the mean distance across all dimensions
    return np.mean(distances)



# 학습률을 동적으로 조정하는 함수
def adjust_learning_rate(optimizer, epoch):
    if epoch >= 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1  # 학습률을 10분의 1로 감소

# 점진적으로 DivideMix 비율을 조정하는 함수
def get_clean_ratio(epoch):
    if epoch < 3:
        return 1.0
    else:
        return 0.8



# 사전 학습된 ResNet18을 특징 추출기로 사용
print("extracting feature.....")
feature_extractor = resnet18(pretrained=True)
feature_extractor.fc = nn.Identity()  # 마지막 fully connected 층 제거
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()




# 학습 과정

# 학습상태를 체크하기 위해 손실값들을 저장합니다
img_list = []
G_losses = []
D_losses = []
iters = 0

# 사용자 설정 변수
user_zscore_threshold = 5  # None이면 자동 계산, 숫자를 입력하면 해당 값 사용
user_loss_ratio = 0.2  # 기본값 20%, 0에서 1 사이의 값


def preprocess_potential_fake_data(combined_dataset, feature_extractor, device, num_selected_fake):
    is_inlier = detect_outliers(combined_dataset, feature_extractor, user_zscore_threshold)
    potential_fake_indices = np.where(~is_inlier)[0]
    selected_fake_indices = np.random.choice(potential_fake_indices, num_selected_fake, replace=False)
    potential_fake_dataset = torch.utils.data.Subset(combined_dataset, selected_fake_indices)
    
    # 데이터를 미리 GPU로 이동
    potential_fake_data = []
    for data in potential_fake_dataset:
        potential_fake_data.append(data[0].to(device))
    
    return torch.stack(potential_fake_data)

# 사전 처리된 potential fake 데이터 생성
print("Preprocessing potential fake data...")
num_selected_fake = int(len(combined_dataset) * 0.1)  # 1% 선택
potential_fake_data = preprocess_potential_fake_data(combined_dataset, feature_extractor, device, num_selected_fake)

print(f"Total potential fake images: {len(potential_fake_data)}")

# 학습 과정
print("Starting Training Loop...")


for epoch in range(num_epochs):
    # 학습률 조정
    adjust_learning_rate(optimizerD, epoch)
    adjust_learning_rate(optimizerG, epoch)

    clean_ratio = get_clean_ratio(epoch)
    
    if epoch >= 3:  # 3 epoch부터 loss 기반 이상치 제거 및 potential fake 데이터 사용
        second_refined_dataset, _ = refine_dataset_by_loss(combined_dataset, netD, device, 0.2)
        dataloader = torch.utils.data.DataLoader(second_refined_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers, pin_memory=True)
        
        print(f"Epoch {epoch}: Using {len(second_refined_dataset)} clean samples out of {len(combined_dataset)}")
        print(f"Using potential fake data from this epoch")
    else:
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers, pin_memory=True)
        print(f"Epoch {epoch}: Using all {len(combined_dataset)} samples")
        print(f"Not using potential fake data yet")
    
    # 한 에폭 내에서 배치 반복
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) D 신경망을 업데이트 합니다
        ###########################
        netD.zero_grad()
        
        # 진짜 데이터로 학습
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 가짜 데이터 생성
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)

        if epoch >= 3:  # 3 epoch부터 potential fake 데이터 사용
            # potential fake 데이터를 배치 크기에 맞게 랜덤 선택
            indices = torch.randperm(potential_fake_data.size(0))[:b_size]
            batch_potential_fake = potential_fake_data[indices]
            
            # 생성된 가짜 이미지와 potential fake 이미지 합치기
            combined_fake = torch.cat([fake, batch_potential_fake], dim=0)
        else:
            combined_fake = fake
        
        # 합쳐진 가짜 이미지로 학습
        label_fake = torch.full((combined_fake.size(0),), fake_label, dtype=torch.float, device=device)
        output = netD(combined_fake.detach()).view(-1)
        errD_fake = criterion(output, label_fake)
        errD_fake.backward()
        D_G_z1 = output[:b_size].mean().item()  # 생성된 이미지에 대한 판별자 출력 평균

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) G 신경망을 업데이트 합니다
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 생성자는 진짜 라벨을 목표로 합니다
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # 훈련 상태를 출력합니다
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 손실값들을 저장해둡니다
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # fixed_noise를 통과시킨 G의 출력값을 저장해둡니다
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


# 학습 루프가 끝난 후
print("Evaluating generated images...")

# 특징 추출기 로드
feature_extractor = load_feature_extractor()

# CelebA 데이터셋만을 위한 DataLoader 생성
celeba_dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=64, shuffle=True, num_workers=workers)

# 실제 CelebA 이미지 준비 (개수 제한)
real_images = next(iter(celeba_dataloader))[0][:1000]  # 최대 1000개 사용

# Fake 이미지 준비
fake_dataloader = torch.utils.data.DataLoader(anime_tensors, batch_size=64, shuffle=True, num_workers=workers)
fake_images = next(iter(fake_dataloader))[:1000]  # 최대 1000개 사용

# 생성된 이미지 준비 (실제 이미지 개수만큼)
num_generated_images = len(real_images)
fake_noise = torch.randn(num_generated_images, nz, 1, 1, device=device)
generated_images = netG(fake_noise).detach()

# 특징 추출
real_features = extract_features(real_images, feature_extractor)
fake_features = extract_features(fake_images, feature_extractor)
generated_features = extract_features(generated_images, feature_extractor)

# 특징 거리 계산
real_vs_generated_distance = calculate_feature_distance(real_features, generated_features)
fake_vs_generated_distance = calculate_feature_distance(fake_features, generated_features)
real_vs_generated_wasserstein = calculate_wasserstein_distance(real_features, generated_features)
fake_vs_generated_wasserstein = calculate_wasserstein_distance(fake_features, generated_features)


print(f"Average Feature Distance (Real vs Generated): {real_vs_generated_distance:.4f}")
print(f"Average Feature Distance (Fake vs Generated): {fake_vs_generated_distance:.4f}")
print(f"Wasserstein Distance (Real vs Generated): {real_vs_generated_wasserstein:.4f}")
print(f"Wasserstein Distance (Fake vs Generated): {fake_vs_generated_wasserstein:.4f}")

# FID 계산
print("Calculating FID...")
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.fc = torch.nn.Identity()
fid_value = calculate_fid(real_images, generated_images, batch_size=32, device=device)
print(f"Final FID (CelebA): {fid_value}")

# 생성된 이미지 시각화
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()