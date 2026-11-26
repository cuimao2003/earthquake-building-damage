# 震后建筑定位和损伤分级重训记录

本轮按建筑级三档宏 F1 查验接近 80% 的目标，同时保留总体正确率、各档精确率和召回率。建筑自动检测采用预测框与人工框的 IoU 至少为 0.50、同一建筑只匹配一次的规则。给定人工框的分级诊断用于判断瓶颈在定位还是伤级判断。墨西哥模型使用震前、震后 RGB 影像，目标为定位建筑并输出分级结果。按现有 xBD 标签分为未受损、轻损、重损或毁坏，后两种严重等级合并训练。旧项目文档中的太阳几何、多光谱、SAR 和结构损失比例需要相应的输入与标注另行验证。

## 数据和划分

| 数据 | 训练 | 验证 | 留出测试 |
|---|---:|---:|---:|
| 墨西哥 xBD 双时相影像 | 130 对 | 27 对 | 34 对 |
| 土耳其 OCD-BDA 人工复核版影像 | 39 对 | 10 对 | 11 对 |

墨西哥留出集来自本机 xView2 的 `hold` 压缩包，此前未用于上一轮训练和测试。原先已看过的测试分区在本轮改作开发数据。切片划分筛除与新留出集有估算地理重叠的开发切片，并检查切片名和建筑 UID。训练、验证、测试的具体切片及计数见 `data/manifest.json` 和 `data/audit.json`。墨西哥训练分区有 157 栋轻损建筑和 42 栋重损或毁坏建筑，留出集分别只有 27 栋和 5 栋。

土耳其的分区以整对影像为单位。通过局部图像特征和几何一致性筛查，将识别到重叠的影像对放入同一分区。其三级定义为屋顶无明显损伤、部分损伤和整体倒塌，使用人工建筑框进行分级评估。它的结果单独报告。来源为 [OCD-BDA 论文](https://www.mdpi.com/2072-4292/16/22/4263) 和 [人工复核数据发布页](https://doi.org/10.5281/zenodo.21296833)。

阿富汗赫拉特数据的标注只明确给出重损和完全倒塌区域。提取 3,889 个连通标注区域作辅助正样本，用灰度统一震前全色影像和震后影像。墨西哥城公开倒塌点对墨西哥训练建筑的检查没有唯一空间匹配。来源为 [MEDBFS 论文](https://www.mdpi.com/2306-5729/9/7/88)、[赫拉特开放数据](https://zenodo.org/records/11241134) 和 [墨西哥城倒塌点图层](https://serviciosatlas.sgirpc.cdmx.gob.mx/arcgis/rest/services/AtlasCapasPublicas/SPC_Eventos/MapServer/0)。

## 实验方法

墨西哥分级对照包括轻量损伤头、DINOv2 冻结特征、关键点配准、按影像的纹理归一化、RBF 分类器、完好外观预测、跨地震迁移和辅助重损样本。轻损和重损的候选模型及阈值由验证集确定。定位候选包括上一轮像素模型区域化、建筑中心点检测和 Swin FPN 框检测。建筑提取训练数据混合使用后，Swin 验证集的建筑框 F1 由 0.5384 升至 0.5524。首轮留出测试后增加了阿富汗辅助训练和 Swin 定位，复测仍用相同留出集，训练和阈值仍由训练与验证分区决定。屋顶局部配准与纹理的对照实验将人工框验证宏 F1 从 0.3900 提到 0.3910，最终配置保留原两支分级器。结果见 `run/mexico_local_change_audit/validation.json`。

随后直接用墨西哥真实训练标注微调了 DINOv2 最后两块，人工框验证宏 F1 为 0.3552。难负样本训练及两支分级器组合的最佳验证宏 F1 约为 0.3910。采集元数据的太阳方位角与高度角用于沿阴影方向提取局部对比特征，最佳组合验证宏 F1 仍约为 0.3910。这些小幅差异未用于再次改变留出配置。对应记录为 `run/mexico_direct_finetune/validation_comparison.json`、`run/mexico_final/last_candidate_audit.json` 和 `run/mexico_solar_geometry/validation.json`。墨西哥训练集采集元数据记录的典型离轴角为震前 19.7°、震后 34.4°，太阳高度角为震前 44.6°、震后 68.7°。采集参数汇总见 `data/development_image_audit.json`。

土耳其模型比较 112 像素和 168 像素的建筑图块，测试了按屋顶变化加权、部分损伤合成增强、成对分类和推理增强。最终在给定人工框的 1,432 栋留出建筑上核验。

参考了 Mexico City 的两阶段分类研究和公开视觉预训练模型。同震例论文的高分使用把未受损与轻损合并的二分类，采用了不同的样本构造。来源为 [两阶段墨西哥地震分类研究](https://www.mdpi.com/2072-4292/14/4/1012) 和 [DINOv2 原始研究](https://arxiv.org/abs/2304.07193)。

## 墨西哥留出复测

| 同一 34 对影像 | 建筑框 F1 | 端到端三档宏 F1 | 人工标注建筑中正确定位且分级 | 判对轻损 | 判对重损 |
|---|---:|---:|---:|---:|---:|
| 旧模型 | 0.2143 | 0.0713 | 25.95% | 0/27 | 0/5 |
| 首轮重训 | 0.3486 | 0.1162 | 41.12% | 1/27 | 0/5 |
| Swin 与双分级器 | 0.4731 | 0.1561 | 47.13% | 0/27 | 0/5 |

复测中有 15 栋轻损和 3 栋重损达到建筑框匹配标准，其等级均被判成未受损。给定人工框时，正确分级 1 栋轻损和 0 栋重损，三档宏 F1 为 0.3382，Cohen’s κ 为 0.0184。整图像素吻合率 82.11%，Kappa 0.5969，轻损与重损的像素宏 F1 为 0.0023。总体像素一致性主要来自背景和未受损区域，建筑损伤分级尚未达到 80% 的目标。完整记录和锁定配置见 `run/mexico_followup_holdout/`。

## 新标注与定位对照

墨西哥城普查图层共有 6,936 条记录，严格匹配到训练屋顶 14 栋。UNAM 两个现场勘查图层分别匹配 2 栋，UNOSAT 五个墨西哥城图层合计匹配 3 栋。普查结构风险、现场结构损伤、卫星可见毁损有各自的标签定义，匹配资料保留在 `data/external/cdmx_census/`。同震次 Copernicus EMSR244 的分级范围在普埃布拉，与当前 xBD 墨西哥城影像范围无重合。来源为 [普查图层](https://rmgir.proyectomesoamerica.org/server/rest/services/Trans_Sept_2018/PlataformaCDMX/MapServer/0)、[CENAPRED 震害数据目录](https://rmgir.proyectomesoamerica.org/ANR/archivo/sismos.html) 和 [EMSR244](https://mapping.emergency.copernicus.eu/activations/EMSR244/)。

摩洛哥的 1,245 个明确受损或毁坏建筑点中，685 个匹配到震前 Open Buildings 屋顶，576 个可提取有效的 Maxar 震前震后光学图块。人工标注为受损 299、毁坏 277。等级由 Copernicus EMS 图层提供，房屋边界用于裁图。受损与毁坏由独立辅助损失使用，确认毁坏的样本也单独测试了重损正样本迁移。样本来源及质量统计位于 `data/external/morocco/`。来源为 [Copernicus EMSR695](https://mapping.emergency.copernicus.eu/activations/EMSR695/)、[Maxar Open Data](https://xpress-docs.maxar.com/OpenData/OpenData_events.htm) 和 [Open Buildings 数据发布页](https://data.humdata.org/dataset/openbuildings_morocco_earthquake_footprint)。

192 像素检测切片微调后在验证集上的建筑框 F1 为 0.5493。256 像素模型原图与翻转图的加权框融合达到 0.5656。以原分级器重新校准后，检测分级合并的验证宏 F1 为 0.2101，原配置为 0.2241。另测试了航拍 SatlasPretrain 特征及模拟自动框偏移的训练增强，给定人工框时各自单模型验证宏 F1 为 0.3596 和 0.3575。来源为 [SatlasPretrain 模型发布页](https://github.com/allenai/satlaspretrain_models)。

## 运行墨西哥影像推理

```powershell
Set-Location -LiteralPath 'D:\土坑论文\earthquake_transformer\retrain'
$quakePython = 'F:\transformer\anaconda\envs\transformer\python.exe'
& $quakePython .\infer_retrain.py image --pre 'F:\transformer\mexico-earthquake_00000131_pre_disaster.png' --post 'F:\transformer\mexico-earthquake_00000131_post_disaster.png' --out .\run\prediction_current
```

输出包括预测框与分级叠加图、以震前影像坐标为基准的栅格分级图、建筑框 CSV 和运行摘要。栅格的 0 为背景，1 为未受损，2 为轻损，3 为重损或毁坏。模型的部件路径、分级阈值和定位版本由 `run/mexico_final/selection.json`、`run/mexico_final/automatic_thresholds.json` 和 `run/localizer_selection.json` 指定。

同一目录中可用下列命令复核墨西哥留出结果，输出到单独的文件夹。

```powershell
& $quakePython .\infer_retrain.py test --with-box-diagnostic --pixel-audit --out .\run\mexico_recheck
```

## 复核土耳其分级

在相同 PowerShell 目录中，可运行以下命令复核锁定的土耳其留出集。它根据验证集已经保存的阈值和权重组合进行推理。

```powershell
& $quakePython .\finetune_turkey.py --test
```

已对实际影像验证单场景推理。输入的建筑框 JSON 格式为 `[[左, 上, 右, 下], ...]`，坐标使用震前和震后配准影像的共同像素坐标。

```powershell
& $quakePython .\finetune_turkey.py --pre .\data\turkey\pre\04.jpg --post .\data\turkey\post\04.jpg --boxes .\run\turkey_box_demo\boxes.json --prediction-dir .\run\turkey_box_demo
```

土耳其留出集的总体正确率为 81.15%，三档宏 F1 为 0.7255，Cohen’s κ 为 0.6907。各档 F1 分别为 A 0.8487、B 0.4359、C 0.8919。完整混淆矩阵见 `run/turkey_vit/test.json`。该实验给定人工建筑框，样例图 `run/turkey_vit/test_examples.png` 同时展示正确结果和误判。

土耳其端到端实验使用同一留出集，把训练分区的人工框用于微调 Swin。使用验证集比较了三个输入尺度及其组合，并锁定分级阈值。留出结果的建筑框 F1 为 0.7828，正确定位且分级为 989/1,432（69.06%），三档宏 F1 为 0.5836。部分损伤正确分级为 67/184，倒塌为 491/671。完整记录见 `run/turkey_automatic/test.json`。

```powershell
& $quakePython .\turkey_automatic.py test
& $quakePython .\turkey_automatic.py image --pre .\data\turkey\pre\04.jpg --post .\data\turkey\post\04.jpg --prediction-dir .\run\turkey_automatic\example_04
```
