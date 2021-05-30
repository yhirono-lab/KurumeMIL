import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def set_LossFunction(dataset, rank):
    # 不均衡データに対してlossの重みを調整
    n_RIGHT = len(dataset[dataset[:,1]=='0'])
    n_LEFT = len(dataset[dataset[:,1]=='1'])
    weights = torch.tensor([1/(n_RIGHT/(n_RIGHT+n_LEFT)), 1/(n_LEFT/(n_RIGHT+n_LEFT))])
    loss_fn = nn.CrossEntropyLoss(weight = weights.to(rank))
    return loss_fn

class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.feature_ex = nn.Sequential(*list(vgg16.children())[:-1])
    
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.view(feature.size(0), -1)
        return feature


class class_predictor(nn.Module):
    def __init__(self, label_count):
        super(class_predictor, self).__init__()
        # 次元圧縮
        self.feature_extractor_2 = nn.Sequential(
            nn.Linear(in_features=25088, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        # attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # class classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, label_count),
        )

    def forward(self, input):
        x = input.squeeze(0)
        H = self.feature_extractor_2(x)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)  # KxL
        class_prob = self.classifier(M)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))

        return class_prob, class_hat, A

class MIL(nn.Module):
    def __init__(self, feature_ex, class_predictor):
        super(MIL, self).__init__()
        self.feature_extractor = feature_ex
        self.class_predictor = class_predictor

    def forward(self, input):
        x = input.squeeze(0)
        # 特徴抽出
        features = self.feature_extractor(x)
        # class分類
        class_prob, class_hat, A = self.class_predictor(features)
        # 訓練時(mode='train')DANN適用
        return class_prob, class_hat, A

'''
ここから下は使用する場合要修正
'''

# 倍率が2つの場合
class MSDAMIL(nn.Module):
    def __init__(self, feature_ex_mag1, feature_ex_mag2, class_predictor):
        super(MSDAMIL, self).__init__()
        self.feature_extractor_mag1 = feature_ex_mag1
        self.feature_extractor_mag2 = feature_ex_mag2
        self.class_predictor = class_predictor
        # 特徴抽出器の計算グラフは不要(更新なし)
        for param in self.feature_extractor_mag1.parameters():
            param.requires_grad = False
        for param in self.feature_extractor_mag2.parameters():
            param.requires_grad = False

    def forward(self, input_mag1, input_mag2):
        mag1 = input_mag1.squeeze(0)
        mag2 = input_mag2.squeeze(0)
        # 各倍率のパッチ画像から特徴抽出
        features_mag1 = self.feature_extractor_mag1(mag1)
        features_mag2 = self.feature_extractor_mag2(mag2)
        # 複数倍率の特徴ベクトルをconcat
        ms_bag = torch.cat([features_mag1, features_mag2], dim=0)
        # class分類
        class_prob, class_hat, A = self.class_predictor(ms_bag, 'test')
        return class_prob, class_hat, A

# 倍率が3つの場合
class MSDAMIL3(nn.Module):
    def __init__(self, feature_ex_mag1, feature_ex_mag2, feature_ex_mag3, class_predictor):
        super(MSDAMIL3, self).__init__()
        self.feature_extractor_mag1 = feature_ex_mag1
        self.feature_extractor_mag2 = feature_ex_mag2
        self.feature_extractor_mag3 = feature_ex_mag3
        self.class_predictor = class_predictor
        # 特徴抽出器の計算グラフは不要(更新なし)
        for param in self.feature_extractor_mag1.parameters():
            param.requires_grad = False
        for param in self.feature_extractor_mag2.parameters():
            param.requires_grad = False
        for param in self.feature_extractor_mag3.parameters():
            param.requires_grad = False

    def forward(self, input_mag1, input_mag2, input_mag3):
        mag1 = input_mag1.squeeze(0)
        mag2 = input_mag2.squeeze(0)
        mag3 = input_mag3.squeeze(0)
        # 各倍率のパッチ画像から特徴抽出
        features_mag1 = self.feature_extractor_mag1(mag1)
        features_mag2 = self.feature_extractor_mag2(mag2)
        features_mag3 = self.feature_extractor_mag3(mag3)
        # 複数倍率の特徴ベクトルをconcat
        ms_bag = torch.cat([features_mag1, features_mag2, features_mag3], dim=0)
        # class分類
        class_prob, class_hat, A = self.class_predictor(ms_bag, 'test')
        return class_prob, class_hat, A
