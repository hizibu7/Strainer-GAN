# final

#celeba + cifar10
#%matplotlib inline
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
import torch.nn.functional as F
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

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



# 커스텀 데이터셋 클래스 정의
class ShuffledCombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum([0] + self.lengths)
        self.length = np.sum(self.lengths)
        self.indices = np.arange(self.length)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        shuffled_idx = self.indices[index]
        dataset_idx = np.searchsorted(self.offsets, shuffled_idx, side='right') - 1
        sample_idx = shuffled_idx - self.offsets[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.length

# CelebA 데이터셋
human_dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# CIFAR-10 데이터셋
cifar10_dataset = dset.CIFAR10(root='./data/cifar-10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

# 데이터셋 크기 출력
human_dataset_size = len(human_dataset)
cifar_dataset_size = len(cifar10_dataset)

print(f"CelebA dataset size: {human_dataset_size}")
print(f"CIFAR-10 dataset size: {cifar_dataset_size}")

# 데이터셋 결합 및 섞기
combined_dataset = ShuffledCombinedDataset(human_dataset, cifar10_dataset)

# DataLoader 정의
dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

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
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(beta1, 0.999))



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




# 학습률을 동적으로 조정하는 함수
def adjust_learning_rate(optimizer, epoch):
    if epoch >= 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1  # 학습률을 10분의 1로 감소

# 점진적으로 DivideMix 비율을 조정하는 함수
def get_clean_ratio(epoch):
    if epoch < 3:
        return 1.0
    elif epoch < 5:
        return 0.8
    elif epoch < 7:
        return 0.6
    else:
        return 0.5



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

# 매 epoch마다 이상치 탐지 및 제거
print("making clean dataset.....")
is_inlier = detect_outliers(combined_dataset, feature_extractor, user_zscore_threshold)
clean_dataset = torch.utils.data.Subset(combined_dataset, np.where(is_inlier)[0])

# 새로운 DataLoader 생성
dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

print(f"Removed {(~is_inlier).sum()} outliers. {len(clean_dataset)} samples remaining.")

# 학습 과정
print("Starting Training Loop...")
first_refined_dataset = clean_dataset  # 초기 정제된 데이터셋 저장


for epoch in range(num_epochs):
    
    # 학습률 조정
    adjust_learning_rate(optimizerD, epoch)
    adjust_learning_rate(optimizerG, epoch)

    noisy_ratio = 1 - get_clean_ratio(epoch)

    if epoch >= 3:  # 3 epoch부터 loss 기반 이상치 제거 적용
        second_refined_dataset, _ = refine_dataset_by_loss(first_refined_dataset, netD, device, 1-noisy_ratio)
        dataloader = torch.utils.data.DataLoader(second_refined_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers)
        
        print(f"Epoch {epoch}: Using {len(second_refined_dataset)} clean samples out of {len(first_refined_dataset)}")
    else:
        print(f"Epoch {epoch}: Using all {len(first_refined_dataset)} samples")
    
    # 한 에폭 내에서 배치 반복
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        ###########################
        ## 진짜 데이터들로 학습을 합니다
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
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



'''
# final

#celeba + cifar10
#%matplotlib inline
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
import torch.nn.functional as F
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve
from sklearn.mixture import GaussianMixture

# 코드 실행결과의 동일성을 위해 무작위 시드를 설정합니다
manualSeed = 1
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



# 커스텀 데이터셋 클래스 정의
class ShuffledCombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum([0] + self.lengths)
        self.length = np.sum(self.lengths)
        self.indices = np.arange(self.length)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        shuffled_idx = self.indices[index]
        dataset_idx = np.searchsorted(self.offsets, shuffled_idx, side='right') - 1
        sample_idx = shuffled_idx - self.offsets[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.length

# CelebA 데이터셋
human_dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# CIFAR-10 데이터셋
cifar10_dataset = dset.CIFAR10(root='./data/cifar-10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

# 데이터셋 크기 출력
human_dataset_size = len(human_dataset)
cifar_dataset_size = len(cifar10_dataset)

print(f"CelebA dataset size: {human_dataset_size}")
print(f"CIFAR-10 dataset size: {cifar_dataset_size}")

# 데이터셋 결합 및 섞기
combined_dataset = ShuffledCombinedDataset(human_dataset, cifar10_dataset)

# DataLoader 정의
dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

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



def evaluate_dataset(netD, dataset, device):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    losses = []
    netD.eval()
    criterion = nn.BCELoss(reduction='none')
    
    with torch.no_grad():
        for data in dataloader:
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            loss = criterion(output, label)
            losses.extend(loss.cpu().numpy())
    
    return np.array(losses)



def divide_dataset(losses, dataset):
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    losses = losses.reshape(-1, 1)
    gmm.fit(losses)
    
    # 두 가우시안 분포의 평균과 표준편차를 구합니다
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    
    # clean 분포(평균이 더 작은 분포)의 인덱스를 찾습니다
    clean_idx = np.argmin(means)
    noisy_idx = 1 - clean_idx
    
    # 두 분포가 만나는 지점을 계산합니다
    a = 1/(2*stds[clean_idx]**2) - 1/(2*stds[noisy_idx]**2)
    b = means[noisy_idx]/(stds[noisy_idx]**2) - means[clean_idx]/(stds[clean_idx]**2)
    c = means[clean_idx]**2 /(2*stds[clean_idx]**2) - means[noisy_idx]**2 /(2*stds[noisy_idx]**2) - np.log(stds[noisy_idx]/stds[clean_idx])
    
    threshold = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    # 임계값보다 loss가 작은 샘플만 clean으로 간주합니다
    clean_idx = losses.flatten() < threshold
    noisy_idx = ~clean_idx
    
    clean_dataset = torch.utils.data.Subset(dataset, np.where(clean_idx)[0])
    noisy_dataset = torch.utils.data.Subset(dataset, np.where(noisy_idx)[0])
    
    return clean_dataset, noisy_dataset


# 학습률을 동적으로 조정하는 함수
def adjust_learning_rate(optimizer, epoch):
    if epoch >= 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1  # 학습률을 10분의 1로 감소

# 점진적으로 DivideMix 비율을 조정하는 함수
def get_clean_ratio(epoch):
    if epoch < 3:
        return 1.0
    elif epoch < 5:
        return 0.5
    elif epoch < 7:
        return 0.7
    else:
        return 0.9



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
user_zscore_threshold = None  # None이면 자동 계산, 숫자를 입력하면 해당 값 사용
user_loss_ratio = 0.2  # 기본값 20%, 0에서 1 사이의 값

# 이상치 탐지 및 제거
print("making clean dataset.....")
is_inlier = detect_outliers(combined_dataset, feature_extractor, user_zscore_threshold)
first_refined_dataset = torch.utils.data.Subset(combined_dataset, np.where(is_inlier)[0])

# 새로운 DataLoader 생성
dataloader = torch.utils.data.DataLoader(first_refined_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

print(f"Removed {(~is_inlier).sum()} outliers. {len(first_refined_dataset)} samples remaining.")

# 학습 과정
print("Starting Training Loop...")


for epoch in range(num_epochs):
    
    # 학습률 조정
    adjust_learning_rate(optimizerD, epoch)
    adjust_learning_rate(optimizerG, epoch)

    if epoch >= 3:  # 3 epoch부터 loss 기반 이상치 제거 적용
        losses = evaluate_dataset(netD, first_refined_dataset, device)
        second_refined_dataset, _ = divide_dataset(losses, first_refined_dataset)
        
        dataloader = torch.utils.data.DataLoader(second_refined_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers)
        
        print(f"Epoch {epoch}: Using {len(second_refined_dataset)} clean samples out of {len(first_refined_dataset)}")
    else:
        print(f"Epoch {epoch}: Using all {len(first_refined_dataset)} samples")
    
    # 한 에폭 내에서 배치 반복
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        ###########################
        ## 진짜 데이터들로 학습을 합니다
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
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
'''


'''
# final

#celeba + cifar10
#%matplotlib inline
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
import torch.nn.functional as F
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

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



# 커스텀 데이터셋 클래스 정의
class ShuffledCombinedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum([0] + self.lengths)
        self.length = np.sum(self.lengths)
        self.indices = np.arange(self.length)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        shuffled_idx = self.indices[index]
        dataset_idx = np.searchsorted(self.offsets, shuffled_idx, side='right') - 1
        sample_idx = shuffled_idx - self.offsets[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.length

# CelebA 데이터셋
human_dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# CIFAR-10 데이터셋
cifar10_dataset = dset.CIFAR10(root='./data/cifar-10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

# 데이터셋 크기 출력
human_dataset_size = len(human_dataset)
cifar_dataset_size = len(cifar10_dataset)

print(f"CelebA dataset size: {human_dataset_size}")
print(f"CIFAR-10 dataset size: {cifar_dataset_size}")

# 데이터셋 결합 및 섞기
combined_dataset = ShuffledCombinedDataset(human_dataset, cifar10_dataset)

# DataLoader 정의
dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

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
optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(beta1, 0.999))



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




# 학습률을 동적으로 조정하는 함수
def adjust_learning_rate(optimizer, epoch):
    if epoch >= 3:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1  # 학습률을 10분의 1로 감소




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

# 매 epoch마다 이상치 탐지 및 제거
print("making clean dataset.....")
is_inlier = detect_outliers(combined_dataset, feature_extractor, user_zscore_threshold)
clean_dataset = torch.utils.data.Subset(combined_dataset, np.where(is_inlier)[0])

# 새로운 DataLoader 생성
dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

print(f"Removed {(~is_inlier).sum()} outliers. {len(clean_dataset)} samples remaining.")

# 학습 과정
print("Starting Training Loop...")
first_refined_dataset = clean_dataset  # 초기 정제된 데이터셋 저장


for epoch in range(num_epochs):
    
    # 학습률 조정
    #adjust_learning_rate(optimizerD, epoch)
    #adjust_learning_rate(optimizerG, epoch)

    if epoch >= 3:  # 3 epoch부터 loss 기반 이상치 제거 적용
        second_refined_dataset, _ = refine_dataset_by_loss(first_refined_dataset, netD, device, user_loss_ratio)
        dataloader = torch.utils.data.DataLoader(second_refined_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=workers)
        
        print(f"Epoch {epoch}: Using {len(second_refined_dataset)} clean samples out of {len(first_refined_dataset)}")
    else:
        print(f"Epoch {epoch}: Using all {len(first_refined_dataset)} samples")
    
    # 한 에폭 내에서 배치 반복
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) D 신경망을 업데이트 합니다: log(D(x)) + log(1 - D(G(z)))를 최대화 합니다
        ###########################
        ## 진짜 데이터들로 학습을 합니다
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## 가짜 데이터들로 학습을 합니다
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) G 신경망을 업데이트 합니다: log(D(G(z)))를 최대화 합니다
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
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
'''