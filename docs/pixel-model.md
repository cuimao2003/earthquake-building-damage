# 墨西哥地震建筑损伤评估

这个最小模型读取同尺寸、已配准的震前和震后 RGB 影像，预测建筑物范围和未受损、轻损、重损或毁坏三个等级。它由共享的双时相编码器、两个四头自注意力 Transformer 块、包含时相差值的多尺度融合层，以及建筑定位和损伤分类两个输出头组成。模型共 196,780 个训练参数，可用单张 GPU 运行。

## 用已训练的权重推理

在 PowerShell 中运行。示例使用本机已有的一对影像，可把两条影像路径替换为自己的配对影像。

```powershell
Set-Location -LiteralPath 'D:\土坑论文\earthquake_transformer'
$quakePython = 'F:\transformer\anaconda\envs\transformer\python.exe'
& $quakePython .\infer.py --pre 'F:\transformer\mexico-earthquake_00000131_pre_disaster.png' --post 'F:\transformer\mexico-earthquake_00000131_post_disaster.png' --out .\run\prediction_131
```

默认从 `run/damage_model.pt` 读取权重、阈值和推理设置。推理使用 256 像素切片和重叠区域加权拼接，输出保持原影像像素尺寸。可用 `--device cpu` 切换到 CPU。

| 文件 | 内容 |
|---|---|
| `damage_mask.png` | 像素编码为 0 背景、1 未受损、2 轻损、3 重损或毁坏 |
| `footprint.png` | 二值建筑范围 |
| `damage_colors.png` | 蓝色表示未受损，黄色表示轻损，红色表示重损或毁坏 |
| `damage_overlay.png` | 叠加在震后影像上的分级预测 |
| `prediction.json` | 影像尺寸、各级像素数和使用的权重 |

## 数据与划分

| 阶段 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| 建筑提取预训练 | 4,096 张 TIFF | 409 张 TIFF |  |
| 地震损伤微调 | 101 对影像 | 19 对影像 | 38 对影像 |

预训练在 `F:\transformer\building train` 的 39,346 对影像与掩膜中按固定随机种子抽样。微调只使用地震目录中的 `mexico-earthquake` 双时相影像。测试集实际位于 `D:\pc\earthquake data\test\test`。核对检查了文件配对、影像和掩膜解码、标签元数据、精确影像重复、建筑 UID 和可估算的地理范围。从原训练目录排除一张与测试切片估计重叠约 37% 的切片，两张切片的标注不足以反推地理范围。实际划分及原始类别计数见 `cache/audit.json`。

xBD 的震后建筑多边形提供未受损、轻损、重损和毁坏标注。训练分区只有 2 栋毁坏建筑，因此把毁坏和重损合并，`un-classified` 建筑只参与定位训练，对损伤损失和指标设为忽略。由于标注中受损建筑很少，训练同时抽取普通图块和包含受损建筑的图块，并结合按类平均的损失。验证集决定损伤分类阈值和最佳权重，测试集用于最终评估。标签定义见 [xBD 数据集论文](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf)。

## 复现实验

本机使用 Anaconda 的 `transformer` 环境，Python 3.8、PyTorch 1.12.1、CUDA 11.3，以及 NumPy、OpenCV、Shapely 和 Pillow。准备脚本使用 OpenCV 解码建筑 TIFF。已生成缓存时可直接开始训练。

```powershell
Set-Location -LiteralPath 'D:\土坑论文\earthquake_transformer'
$quakePython = 'F:\transformer\anaconda\envs\transformer\python.exe'
& $quakePython .\prepare.py building --building-limit 4096
& $quakePython .\prepare.py earthquake
& $quakePython .\train.py building --epochs 10 --batch 16 --seconds 1600
& $quakePython .\train.py earthquake --epochs 72 --batch 8 --steps 132 --seconds 5200
& $quakePython .\select_checkpoint.py .\run\damage_best.pt
& $quakePython .\evaluate.py
```

训练脚本按给定时限保存验证分数最高的权重，并记录逐轮损失和定期验证结果。终止条件触发后会完成本轮验证和保存。可用 `--init` 从已保存的震害权重继续微调，用 `--focus` 启用受损建筑周围的变尺度采样和分层损失。选择脚本只用验证集比较候选权重，并校准损伤阈值。测试记录包含逐切片 CSV、逐类别像素指标和基于人工标注轮廓的建筑级诊断。建筑级诊断仅在评估时用标注关联预测像素。

本轮范围是双时相 RGB 建筑定位与损伤分级。独立阴影掩膜、多光谱和 SAR 融合、结构完整度损失百分比估计均超出这版的输入和标注范围。现有测试覆盖同一次墨西哥地震，无法据此评价跨地震的泛化能力。

## 按建筑分级

逐像素模型完成第一次测试后，另训练了一个 61,939 参数的 Transformer 作按建筑分级实验。它以定位图为输入生成建筑区域，在每个区域中心提取震前、震后图块，并把区域分级结果投回同一影像。区域模型训练使用训练分区的建筑标注，实际推理和复测从定位图自动生成区域。两次测试使用同一批 38 对影像，第二次是在看到首轮结果后进行的复测。两次均用训练集拟合参数，用验证集选择权重和阈值。

在上面的 PowerShell 环境中，用以下命令运行按建筑分级。两份权重一起使用，输出仍为相同尺寸的建筑范围和损伤分级图。

```powershell
& $quakePython .\infer.py --pre 'F:\transformer\mexico-earthquake_00000131_pre_disaster.png' --post 'F:\transformer\mexico-earthquake_00000131_post_disaster.png' --region-checkpoint .\run\region_final\region_model.pt --out .\run\region_final\prediction_131
```

按建筑分级的训练和复测入口如下。已有的最佳早期权重位于 `run/regions/region_model.pt`，续训也可用 `--init` 从该权重开始。

```powershell
& $quakePython .\train_regions.py --epochs 42 --steps 88 --batch 48 --seconds 1900
& $quakePython .\train_regions.py --calibrate --init .\run\regions\region_model.pt --out .\run\region_final
& $quakePython .\evaluate.py --region-checkpoint .\run\region_final\region_model.pt --out .\run\region_final --batch 8
```

## 本次测试结果

以下指标基于 38 对测试影像的有效像素。两种方案使用同一张建筑定位图。

| F1 指标 | 逐像素模型 | 按建筑分级复测 |
|---|---:|---:|
| 建筑定位 | 0.7832 | 0.7832 |
| 未受损 | 0.7534 | 0.7341 |
| 轻损 | 0.0000 | 0.0007 |
| 重损或毁坏 | 0.0082 | 0.0233 |
| 任意损伤 | 0.0231 | 0.0345 |

建筑定位 IoU 为 0.6437。在基于标注轮廓的建筑级诊断中，84 栋轻损建筑都没有得到正确等级，32 栋重损或毁坏建筑中两版分别正确分级 2 栋和 1 栋。建筑定位和分级图输出流程已验证，损伤分级精度仍不足以可靠判读。

逐像素结果保存于 `run/test_metrics.json`，按建筑分级的复测结果保存于 `run/region_final/test_metrics.json`。两份结果目录中的 `test_by_tile.csv` 记录逐切片结果，`examples` 包含震前、震后、标注和预测的对照图。样例按人工标注中的损伤数选取。
