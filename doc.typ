#set page(
  paper: "a4",
  margin: (top: 1.5cm, bottom: 1.5cm, left: 2.0cm, right: 2.0cm),
)
#set document(title: "", author: "Ling Yu", date: datetime.today())
#set heading(numbering: "1.1.1")
#let font = (
  main: "Helvetica",
  cjk: "PingFang SC",
  mono: "Maple Mono NF",
  serif: "Times New Roman",
  sans: "Arial",
)
#set text(font: (font.main, font.cjk), lang: "zh")
#show raw.where(block: false): it => (
  text(font: font.mono, it)
)
#show raw.where(block: true): it => block(
  text(font: font.mono, it),
)

= Partition-Aggregation
隐藏层对样本空间做分割，最后一层对分割块进行聚合，聚合成类别数

前几层在生成一个复杂的空间分割（partition / tessellation）

最后一层在这些分割单元上做 class-wise aggregation（线性判别）

$
  f(x) = W phi(x),
  
$
$phi : bb(R)^d arrow R^m, W in bb(R)^(C times m)$

网络在训练过程中，逐层把同类样本的分割块“对齐 / 合并”到同一输出方向

==
一个训练良好的分类网络，会诱导一个输入空间分割，使得

同类样本主要落在少量连通的分割子区域中

输出层对这些区域做近似常数的 class-wise 聚合


== Partition-Aggregation Consistency Regularization
version 1
$ cal(L)_c = 1 / N_c sum_(i:y_i=c) || z_i -mu_c||_2^2 $
```py
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
```



version 3
$ cal(L)_c = ((|| f(z_i)-f(z_j) || )/( || z_i -z_j || + delta))^2 $
```py

def pacr_loss_lipschitz_fast(features, logits, labels):
    eps = 1e-6

    # pairwise distances
    dz = torch.cdist(features, features, p=2)
    dy = torch.cdist(logits, logits, p=2)

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = torch.triu(label_eq, diagonal=1)

    ratio = dy / (dz + eps)
    loss = (ratio[mask] ** 2).mean()

    return loss

```