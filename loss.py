import torch
import torch.nn.functional as F

def pacr_loss(features, labels):
    """
    features: (B, D)
    labels:   (B,)
    """
    loss = 0.0
    num_classes = labels.max().item() + 1
    eps = 1e-8

    for c in range(num_classes):
        idx = labels == c
        if idx.sum() < 2:
            continue
        z = features[idx]  # (Nc, D)
        mean = z.mean(dim=0, keepdim=True)
        loss += ((z - mean) ** 2).sum(dim=1).mean()

    return loss / (num_classes + eps)


def pacr_pairwise_loss(features, labels, num_pairs=1):
    """
    Pairwise PACR-V1 (center-free)

    features: (B, D)
    labels:   (B,)
    num_pairs: number of random pairs per sample
    """
    loss = 0.0
    cnt = 0
    B = features.size(0)

    for i in range(B):
        same = (labels == labels[i]).nonzero(as_tuple=False).squeeze()
        if same.numel() < 2:
            continue

        for _ in range(num_pairs):
            j = same[torch.randint(0, same.numel(), (1,))].item()
            if i == j:
                continue
            loss += (features[i] - features[j]).pow(2).sum()
            cnt += 1

    if cnt == 0:
        return torch.tensor(0.0, device=features.device)

    return loss / cnt


def pacr_pairwise_loss_vec(features, labels):
    """
    Fully vectorized pairwise PACR
    features: (B, D)
    labels:   (B,)
    """
    B, D = features.shape
    loss = 0.0
    cnt = 0

    # pairwise squared distances: (B, B)
    dist2 = torch.cdist(features, features, p=2).pow(2)

    # same-class mask (B, B)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)

    # remove diagonal
    same.fill_diagonal_(False)

    if same.any():
        loss = dist2[same].mean()
    else:
        loss = torch.tensor(0.0, device=features.device)

    return loss


def pacr_pairwise_sampled(features, labels, num_pairs=1):
    """
    Vectorized random-pair PACR
    """
    B, D = features.shape

    # (B, B) same-class mask
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same.fill_diagonal_(False)

    # for each i, sample j from same-class indices
    loss = 0.0
    cnt = 0

    for _ in range(num_pairs):
        # random j for all i
        rand = torch.randint(0, B, (B,), device=features.device)
        valid = same[torch.arange(B), rand]

        if valid.any():
            diff = features[valid] - features[rand[valid]]
            loss += diff.pow(2).sum(dim=1).mean()
            cnt += 1

    if cnt == 0:
        return torch.tensor(0.0, device=features.device)

    return loss / cnt


def pacr_region_logit(features, logits, labels):
    probs = torch.softmax(logits, dim=1)
    conf = probs.max(dim=1).values.detach()  # (B,)

    dist2 = torch.cdist(features, features).pow(2)

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    weight = conf.unsqueeze(0) * conf.unsqueeze(1)

    mask = same_class & (weight > 0)

    if mask.any():
        return (dist2 * weight)[mask].mean()
    else:
        return torch.tensor(0.0, device=features.device)


def pacr_region_knn(features, labels, k=5):
    B = features.size(0)
    dist = torch.cdist(features, features)
    knn = dist.topk(k + 1, largest=False).indices[:, 1:]

    loss = 0.0
    cnt = 0

    for i in range(B):
        for j in knn[i]:
            if labels[i] == labels[j]:
                loss += (features[i] - features[j]).pow(2).sum()
                cnt += 1

    if cnt == 0:
        return torch.tensor(0.0, device=features.device)
    return loss / cnt


def pacr_tangential_loss(
    features,  # (B, D)
    logits,  # (B, C)
    labels,  # (B,)
    temperature=10.0,
    eps=1e-8,
):
    """
    Tangential / Directional PACR (Region-aware, center-free)

    - features: penultimate layer features
    - logits:   classifier outputs
    - labels:   ground-truth labels
    """

    B, D = features.shape
    device = features.device

    # -------- 1. compute gradient direction g_i = ∂ logit_y / ∂ z --------
    selected_logits = logits[torch.arange(B), labels]  # (B,)

    grads = torch.autograd.grad(
        outputs=selected_logits.sum(),
        inputs=features,
        create_graph=False,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (B, D)

    # normalize direction (important!)
    grads = grads / (grads.norm(dim=1, keepdim=True) + eps)
    grads = grads.detach()  # critical: no second-order backprop

    # -------- 2. pairwise direction similarity --------
    # cosine similarity in [-1, 1]
    cos_sim = torch.matmul(grads, grads.t())

    # convert to soft region weight in [0,1]
    weight = torch.exp(temperature * cos_sim)

    # -------- 3. same-class mask --------
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)

    # -------- 4. feature pairwise distance --------
    dist2 = torch.cdist(features, features).pow(2)

    mask = same_class & (weight > 0)

    if mask.any():
        return (weight * dist2)[mask].mean()
    else:
        return torch.tensor(0.0, device=device)


def pacr_tangential_light(features, logits, labels, temperature=10.0):
    probs = F.softmax(logits, dim=1)
    conf = probs[torch.arange(len(labels)), labels].detach()

    # 使用 logit 差分方向作为 proxy
    dir_feat = F.normalize(features, dim=1)
    cos_sim = torch.matmul(dir_feat, dir_feat.t())

    weight = torch.exp(temperature * cos_sim) * conf.unsqueeze(0) * conf.unsqueeze(1)

    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    dist2 = torch.cdist(features, features).pow(2)

    mask = same_class & (weight > 0)
    return (weight * dist2)[mask].mean()
