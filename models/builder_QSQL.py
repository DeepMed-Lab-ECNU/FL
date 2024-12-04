
import torch
import torch.nn as nn

def vectorized_triplet_loss(anchors, positives, negatives, margin):
    # 使用广播计算锚点与正、负样本之间的差异
    pos_diff = anchors - positives  # [n, dim]
    neg_diff = anchors.unsqueeze(1) - negatives  # [n, 1, dim] - [n, k, dim] => [n, k, dim]

    # 计算平方和而不是直接使用norm，减少开根号的计算
    pos_dist_squared = torch.sum(pos_diff * pos_diff, dim=1, keepdim=True)  # [n, 1]
    neg_dist_squared = torch.sum(neg_diff * neg_diff, dim=2)  # [n, k]

    # 计算三元组损失，避免使用relu的外部调用，直接在公式中应用最大值函数
    losses = torch.clamp(pos_dist_squared - neg_dist_squared + margin, min=0.0)  # [n, k]

    # 对所有负样本的损失求平均
    loss_mean = torch.mean(losses)  # 直接对所有元素求平均，而不是先按维度

    return loss_mean



class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SELayer, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.LayerNorm(in_channels),  
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Global average pooling
        x_se = self.global_avg_pooling(x)
        x_se = x_se.view(x_se.size(0), -1)

        # Fully-connected layers
        x_se = self.fc(x_se)

        x_se = x_se.view(x.size(0), x.size(1), 1, 1)

        # Element-wise multiplication with input x
        x = x * x_se

        return x







class QSQLModel(nn.Module):


    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, total_images =304):

        super(QSQLModel, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        feature_storage = torch.randn(total_images, 16, dim)
        self.feature_storage = nn.functional.normalize(feature_storage, dim=2)
        
        self.selayer = SELayer(60, 7)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False  


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    
    
    
    
    @torch.no_grad()
    def select_neg_keys(self, q_indices, image_indices):
        """
        为每个q选择负样本k。
        参数:
        - q_indices: 当前批次中q特征的维度索引列表,指示每个q对应的原图中的维度位置。
        - image_indices: 当前批次中每个q对应的图像全局索引。

        返回:
        - neg_keys: 选定的负样本k的集合,形状为(q数量, 选定负样本数量, dim)。
        """
        neg_keys = []
        for q_idx in q_indices:
            # 计算当前q对应的图像全局索引
            global_idx = image_indices[q_idx // 16]  # 由于每张图对应16个q，这里使用q_idx来计算它所在的图像索引

            # 为每个q构建除了其自身维度和配对k维度之外的维度索引
            all_indices = set(range(16))  # 假设每张图有16个维度
            # q_idx % 16 计算出q在其对应图像中的局部索引
            excluded_indices = {q_idx % 16, (q_idx % 16 + 1) % 16}  # 排除当前q维度和直接配对的k维度
            valid_indices = list(all_indices - excluded_indices)

            # 从feature_storage中选择当前q对应图像的有效负样本k
            selected_neg_keys = self.feature_storage[global_idx, valid_indices, :]
            neg_keys.append(selected_neg_keys)

        # 将列表转换为Tensor，形状为(q数量, 选定负样本数量, dim)
        neg_keys = torch.stack(neg_keys, dim=0)
        return neg_keys

    
    @torch.no_grad()
    def select_cross_image_neg_keys(self, q_indices, image_indices):
        """
        为每个q选择与之处于相同维度的其他图像的负样本k。
        参数:
        - q_indices: 当前批次中q特征的维度索引列表,指示每个q对应的原图中的维度位置。
        - image_indices: 当前批次中每个q对应的图像全局索引。

        返回:
        - cross_neg_keys: 跨图像选定的负样本k的集合,形状为(q数量, 图像数量-1, dim)。
        """
        cross_neg_keys = []
        total_images = self.feature_storage.shape[0]  # 假设总图像数为304
        for q_idx in q_indices:
            global_idx = image_indices[q_idx // 16]  # 计算q对应的图像全局索引
            q_dim = q_idx % 16  # 计算q在其图像中的维度位置
            
            # 对于每个q，选择除了当前图像之外的所有图像的指定维度特征
            selected_neg_keys = [self.feature_storage[i, q_dim, :] for i in range(total_images) if i != global_idx]
            selected_neg_keys = torch.stack(selected_neg_keys, dim=0)  # 将列表转换为Tensor
            cross_neg_keys.append(selected_neg_keys)

        # 将列表转换为Tensor，形状为(q数量, 图像数量-1, dim)
        cross_neg_keys = torch.stack(cross_neg_keys, dim=0)
        return cross_neg_keys


    
    

    def forward(self, x, image_indices):
        """
        Input:
            x: 输入批次的图像数据
            image_indices: 当前批次中每个图像的全局索引
        Output:
            logits, targets
        """
        device = x.device  
        actual_batch_size = x.size(0)
        x = self.selayer(x)

        x = x.view(-1, 3, 256, 256)
        x = torch.stack([x[j] for j in range(actual_batch_size*20) if j % 20 not in {0, 1, 18, 19}], dim=0)
        x = x.view(actual_batch_size, 16, 3, 256, 256)

        # 分离查询图像和键图像
        im_q = x[:, ::2, :, :, :]
        im_k = x[:, 1::2, :, :, :]

        im_q = im_q.contiguous().view(-1, 3, 256, 256)
        im_k = im_k.contiguous().view(-1, 3, 256, 256)

        # 计算q和k特征
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        
        # 准备q对应的维度位置
        q_indices = [i*2 for i in range(actual_batch_size*8)]  # 偶数位置对应q
        # 对于每个q，选择与之对应的负样本k
        neg_keys = self.select_neg_keys(q_indices, image_indices)  # 更新select_neg_keys调用
        neg_keys = neg_keys.to(device)

        
        
        # 选择与q处于相同维度的其他图像的负样本
        cross_neg_keys = self.select_cross_image_neg_keys(q_indices, image_indices)
        cross_neg_keys = cross_neg_keys.to(device)  # 确保负样本在正确的设备上
        # 定义距离超参数
        margin_neg = 1.0  
        margin_cross_neg = 0.3  

        # 计算两组负样本损失
        loss_neg = vectorized_triplet_loss(q, k, neg_keys, margin_neg)
        loss_cross_neg = vectorized_triplet_loss(q, k, cross_neg_keys, margin_cross_neg)
        total_loss = loss_neg + loss_cross_neg

        with torch.no_grad():  # no gradient to keys


            k_second = self.encoder_k(im_q)  # keys: NxC
            k_second = nn.functional.normalize(k_second, dim=1)
            # 更新特征存储结构，正确存储k和交换后的k特征
            for idx, global_idx in enumerate(image_indices):
                # print(global_idx)
                offset = idx * 8  # 计算当前批次开始的偏移量
                # 存储最初计算的k特征到奇数位置
                for i in range(8):
                    self.feature_storage[global_idx, i*2 + 1, :] = k[offset + i, :]
                # 存储交换后的k特征到偶数位置
                for i in range(8):
                    self.feature_storage[global_idx, i*2, :] = k_second[offset + i, :]

        return total_loss



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [tensor.clone()]
    output = torch.cat(tensors_gather, dim=0)
    return output
