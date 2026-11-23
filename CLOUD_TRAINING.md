# 眼科数据集云端训练操作说明
## 当前DDR迁移实验

用户已选择先尝试APTOS权重再训练DDR，效果不佳时才考虑原始预训练对照。当前配置为 `configs/experiments/dr_vitl_ddr_from_aptos_head_20261113.json`，从APTOS第5轮 `head_best.pt` 导入模型权重，SHA256固定为 `80c13752f5655290b2349ebc1dce951c83ef7aa7aa27d17572d09fbd0078fe05`。这是APTOS→DDR顺序迁移：本轮训练清单仅DDR，但模型监督学习历史包含APTOS和DDR，不称为独立DDR-only实验。

初始化严格验证父模型配置、标签、张量与文件SHA，只导入模型参数；优化器、调度器、scaler、轮次和RNG重新开始。完整DDR验证集2,479张先于任何更新评估，初始QWK为0.6411719，真实3/4级225张中42张预测为0/1级。后续同时报告验证QWK和这类严重低估；外部测试不参与逐轮调参。

训练6,041张、验证2,479张、内部测试3,898张，IDRiD506条仅作固定外部来源基准。DDR不可判读标签不进入0–4级分类。训练继续使用ViT-L/14、336、FP16、有效batch32、前5轮分类头及后25轮最后4个模块。原始预训练独立对照配置 `dr_vitl_ddr_20261113_fast.json` 已保留，尚未启动。

APTOS补充审计对3,488张全部重算文件和完整RGB像素SHA，跨划分精确重复为0；原固定复合近重复规则命中0，测试相关12个更宽哈希候选没有确认同眼重复，仍有不确定风险且患者ID缺失。用户已接受现有划分继续训练，数据和划分保持原样。完整审计见 `cloud_upload/aptos_vitl_20261113/leakage_audit/audit_summary.json`，不将这些检查表述为保证所有近重复或患者关联绝对为0。

## 当前APTOS独立实验

用户当前要求先评估APTOS，DDR暂缓。实际配置为 `configs/experiments/dr_vitl_aptos_20261113_fast.json`：DINOv2-L/14预训练、336输入、30轮；前5轮只训练分类头，后25轮仅训练最后4个Transformer模块及分类头，整个实验不解冻全主干。主干学习率1e-5，分类头1e-3。两张V100每卡batch16、累积1步，有效batch32，FP16；已在新批量下探测显存后关闭梯度检查点重算。

只用APTOS训练2,775张、验证359张，验证QWK选最佳；锁定检查点后评估APTOS测试354张与IDRiD外测506张。IDRiD已经用于历史实验外测，不再称为从未观察过的盲测集，禁止据其调参。保存前5轮的最佳分类头参考与最终全程最佳，旧混合来源DINOv2-S模型保持原样。旧混合验证总分不能直接与APTOS验证分数比较；对照须使用相同APTOS测试子集。

缓存从原始APTOS图片独立生成，只含train/val，336缩放后的PNG逐图校验。入口默认尊重配置中的workers；`TRAIN_WORKERS`可显式覆盖。正式实验示例：

```bash
bash scripts/launch_v100.sh configs/experiments/dr_vitl_aptos_20261113_fast.json /mnt/retina/runs/dr_vitl_aptos_20261113_fast
```

## 当前单来源规则（2026-11-13）

新实验先按协议、任务、数据来源隔离，再选择模型。旧训练和备份保持原样；旧混合配置不能启动新的正式训练或续训，仍可用于读取历史检查点做评估。下文存储统计为旧包参考，新包大小必须重新用打包工具统计。

- 默认DR仅使用DDR：训练6,041，验证2,479，内部测试3,898。APTOS独立对照为2,775 / 359 / 354，两个来源不拼接。
- EyeQ默认只用于图像质量任务；其真实DR等级来自EyePACS，原标签不删除、不改写，不把质量等级当DR等级。DDR不可判读与EyeQ、FIVES质控各自保留输出定义。
- 单来源配置的外测只能通过`data.external_dataset`明确指向一份独立来源清单。当前严格协议DR配置将IDRiD的506条记录仅用于外测。
- `strict_cross_domain`与`within_source`分别管理。FIVES血管/质控和IDRiD定位的来源内实验，不能与严格协议专家拼成一个声称IDRiD/FIVES全未见的系统外测。
- 患者、内容哈希和原划分标识按已有元数据检查；缺失患者身份时，不声称完整临床患者独立。

隔离目录为`data/prepared/source_isolated/<协议>/tasks/<任务>/<来源>/<split>.jsonl`，每条原记录的字节、原图相对路径、原标签和split均保留。原始目录不移动、不重划。

```bash
# 首次构建；已有同身份目录只读校验复用，冲突则拒绝覆盖
python scripts/isolate_training_datasets.py --data-root data --protocol both --output-root data/prepared/source_isolated
python scripts/generate_source_configs.py --include-idrid-external

# 只检查一个拟议专家组合，不启动训练
python scripts/check_expert_suite.py --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json --config configs/source_isolated/strict_cross_domain/image_quality_eyeq_eyeq.json --data-root data
```

新分类权重只从对应来源的训练标签重新计算。缓存也必须绑定这份单来源train/val清单；旧混合缓存不能直接用于新实验。模型大小和训练超参暂沿用模板，DINOv3-L仅为待验证候选，本次没有启动新训练。


本目录提供训练入口、固定数据划分、任务配置和上传工具。当前服务器方案为两张 Tesla V100，每张16GB显存，16vCPU和128GB内存。默认先训练DR五级分类，使用双卡数据并行；其他任务分别启动，不把异构标签混在同一个分类头中。

数据准备完成不等于模型已训练。上传前的本地检查和CPU小样本运行只能验证程序链路；完整训练时长、显存峰值和模型指标需在云服务器实测。

## 磁盘和上传目录

系统盘40GB仅适合作为系统和基础环境盘。数据、模型下载缓存和训练输出放在单独的数据盘，例如 `/mnt/retina`。先用下面的命令查看所选任务的实际上传大小，再分配数据盘；压缩包上传后解压会同时占用归档和解压数据两份空间，还要预留环境、权重缓存及检查点空间。

历史混合DR实验全部划分所需文件约 **20.64GiB**。按未压缩大小计算，归档与解压数据同时保留约需41.27GiB，已超过40GiB系统盘。DR实验建议单独准备至少100GiB数据盘；压缩率和检查点数量会影响实际占用。

历史13个模板所需文件的去重并集约 **34.81GiB**，不含预训练权重和训练输出。若准备开展全部任务和大模型对照，建议数据盘200GiB，以保留归档、模型缓存和多个实验的最近/最优检查点。

在本地项目根目录运行：

```powershell
# 只统计默认DR任务，不创建大归档
python scripts/package_cloud_training.py --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json

# 统计全部配置所需文件的去重并集
python scripts/package_cloud_training.py --all-configs

# 纯代码包，适合已通过其他方式上传完整data目录的情况
python scripts/package_cloud_training.py --code-only --output cloud_training_code.zip

# DR任务的代码、清单和必需影像；此命令会实际读取并打包全部选中数据
python scripts/package_cloud_training.py --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json --output dr_training.tar.gz
```

可重复使用 `--config` 同时包含多个任务。归档保留 `data/` 下的相对路径，不移动或改写源图、不重新划分数据。工具默认只预览，提供 `--output` 后才创建归档；已有同名归档会拒绝覆盖。

任务包包含所选任务的训练、验证、内部测试和外部测试清单，以及它们引用的图像、训练用PNG掩膜和检测XML。原始压缩包、提交材料、中间产物和大型指纹数据库不进入任务包。`source_path` 指向的原始标注只作审计，不是训练时必须读取的文件。小型来源说明会随数据保留。

归档内的数据条目保留清单引用的大小写，以兼容Linux区分大小写的文件系统。本地对全部配置的72,974个唯一运行时引用逐一检查，未发现引用拼写与实际文件大小写不一致。

如果上传完整数据目录，至少保持以下结构，不要只上传 `data/prepared`：

```text
retina/
  training/
  configs/
  scripts/
  requirements-training.txt
  CLOUD_TRAINING.md
  data/
    prepared/
      tasks/...
      protocols/within_source/tasks/...
      assets/masks/...
    APTOS2019/...
    DDR/...
    EyeQ/...
    ...其他选中配置引用的来源
```

任务子集包没有完整的10任务目录，不能用全协议的 `validate_training_sets.py` 判断上传是否完整；应运行下文的 `cloud_preflight.py`。新训练包使用物理分开的单来源清单；`data.datasets` 必须恰好声明一个训练来源。发现异来源记录时直接报错。原混合清单仅保留作历史实验审计和旧模型评估。

## 云端环境

使用Linux、Python3.10或3.11，并为V100安装支持其CUDA计算能力的PyTorch版本。不要把本机Anaconda环境复制到Linux。本机当前PyTorch为CPU构建，即使机器有GPU，也不代表该环境能运行CUDA。

已租用服务器使用Ubuntu22.04 LTS；平台驱动说明见华为云的[GPU加速型ECS驱动安装说明](https://support.huaweicloud.com/usermanual-ecs/ecs_03_0199.html)确认驱动可用。下面的环境脚本负责Python环境，不代替平台GPU驱动安装。

在云服务器新建环境，将项目解压到数据盘：

也可使用随包提供的脚本：在项目根目录运行 `bash scripts/setup_cloud.sh` 创建 `/mnt/retina/venv` 并安装依赖；运行 `bash scripts/launch_v100.sh` 启动默认双卡DR训练。`RETINA_RUNTIME_ROOT` 可指定其他数据盘目录，`RETINA_DATA_ROOT` 可指定影像目录。脚本不会挂载或格式化磁盘，使用前应确保路径确实位于已挂载的数据盘上。下面保留对应的逐步命令。

```bash
mkdir -p /mnt/retina/project /mnt/retina/cache /mnt/retina/runs
cd /mnt/retina/project
tar -xzf /path/to/dr_training.tar.gz
python3 -m venv /mnt/retina/venv
source /mnt/retina/venv/bin/activate
python -m pip install --upgrade pip

# CUDA版PyTorch与torchvision必须成对安装；下列V100组合应与平台驱动兼容
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-training.txt

export HF_HOME=/mnt/retina/cache/huggingface
export TORCH_HOME=/mnt/retina/cache/torch
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

上述安装示例要求平台已有兼容的NVIDIA驱动。若镜像已预装另一套可用环境，先测试GPU识别和训练，再决定是否更换依赖，避免同时混用多套torch/torchvision。CUDA版wheel自带所需用户态CUDA运行库，不要求另装完整CUDA Toolkit才能训练。V100训练使用 `fp16`，不要配置 `bf16`。

安装版本对照见 [PyTorch官方历史版本表](https://docs.pytorch.org/get-started/previous-versions/)。

模型配置若使用预训练主干，首次创建模型可能下载权重。把缓存放到数据盘，并确认服务器可以访问权重服务；离线机器需预先准备匹配的权重和缓存。没有取得权重时不要把随机初始化实验标成预训练实验。

## 开训前检查

```bash
python scripts/cloud_preflight.py \
  --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json \
  --data-root /mnt/retina/project/data

# 单卡、真实样本的小批量前后向和验证，用于验证代码路径
python -m training.train \
  --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json \
  --data-root /mnt/retina/project/data \
  --output /mnt/retina/runs/dr_smoke \
  --device cuda --workers 0 --smoke-test
```

预检检查选定任务的文件引用、训练及验证监督、划分一致性、按来源抽样解码和CUDA环境。其默认要求CUDA可用；本地仅调试可增加 `--allow-cpu`。预检不会下载模型或安装依赖，也不会重新计算全部图像哈希。首次实际训练前仍需上述小批量运行。

## 双卡训练

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m training.train \
  --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json \
  --data-root /mnt/retina/project/data \
  --output /mnt/retina/runs/dr_ddr_isolated \
  --device cuda --workers 4
```

`--workers 4` 指每个训练进程4个数据加载进程，双卡合计8个。先同时观察GPU利用率、显存和CPU负载再调整。配置中的batch size以每张卡为单位；有效批量等于每卡batch size × 卡数 × 梯度累积步数。显存不足时先减小每卡batch size，并通过梯度累积保持所需有效批量。

默认DR配置的每卡batch size为4，梯度累积4步，双卡有效批量为32；模型为DINOv2-S，输入392×392。DINOv3-L另有独立配置，用更小的每卡批量，显存占用仍需实测。

输出目录应按任务、协议和实验命名。只根据验证集选择最优模型，内部测试和外部测试不参与训练、调参或阈值选择。日常运行不要自动遍历所有任务；先完成DR基线，再分别启动其他配置。

使用单卡时直接运行 `python -m training.train ...`。长任务可在 `tmux` 中启动，防止SSH断连终止训练。

## 断点恢复和独立测试

训练会保存最近和最优检查点，恢复时指定原实验的最近检查点。检查点文件名及最终指标文件以训练程序输出为准：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m training.train \
  --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json \
  --data-root /mnt/retina/project/data \
  --output /mnt/retina/runs/dr_ddr_isolated \
  --resume /mnt/retina/runs/dr_ddr_isolated/last.pt \
  --device cuda --workers 4

# 内部测试，显式指定最优检查点
python -m training.train \
  --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json \
  --data-root /mnt/retina/project/data \
  --output /mnt/retina/runs/dr_test \
  --eval-only --checkpoint /mnt/retina/runs/dr_ddr_isolated/best.pt \
  --split test --device cuda --workers 4

# 跨来源外测另存结果
python -m training.train \
  --config configs/source_isolated/strict_cross_domain/dr_grading_ddr.json \
  --data-root /mnt/retina/project/data \
  --output /mnt/retina/runs/dr_external \
  --eval-only --checkpoint /mnt/retina/runs/dr_ddr_isolated/best.pt \
  --split external_test --device cuda --workers 4
```

恢复训练时保持任务、协议、标签顺序和模型结构一致。更换模型或划分属于新实验，应使用新输出目录。

`--resume` 的输出目录必须是原检查点所在目录，以保留历史记录和最优检查点。需要计划暂停时可使用 `--stop-after-epoch 1`，它会在完成第一轮后保存并退出，不改变原配置的总轮数和学习率计划；后续用原配置及 `--resume .../last.pt` 继续。中途在某轮内停止时，从最近完成的epoch恢复。

每个实验目录保存 `config.resolved.json`、`runtime.json`、`data_identity.json`、`history.jsonl` 和检查点；显式评估生成 `evaluation_test.json` 或 `evaluation_external_test.json`。检查点包含优化器、调度器、混合精度和随机状态，按epoch边界恢复；恢复时卡数、批量、累积步数、总epoch和学习率计划需保持一致。

## 可选任务配置

以下命令中的 `--config` 可以替换为对应文件。每个任务单独建立输出目录，不同时把全部实验挤进双卡显存。

| 单来源配置 | 任务 / 来源 | 协议 |
|---|---|---|
| `configs/source_isolated/strict_cross_domain/disc_cup_segmentation_papila.json` | disc_cup_segmentation / PAPILA | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/disc_cup_segmentation_refuge.json` | disc_cup_segmentation / REFUGE | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/dr_grading_aptos2019.json` | dr_grading / APTOS2019 | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/dr_grading_ddr.json` | dr_grading / DDR | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/dr_grading_dinov3_large_aptos2019.json` | dr_grading / APTOS2019 | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/dr_grading_dinov3_large_ddr.json` | dr_grading / DDR | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/glaucoma_classification_papila.json` | glaucoma_classification / PAPILA | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/image_quality_ddr_ddr.json` | image_quality / DDR | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/image_quality_eyeq_eyeq.json` | image_quality / EyeQ | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/lesion_detection_ddr.json` | lesion_detection / DDR | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/lesion_segmentation_ddr.json` | lesion_segmentation / DDR | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/multidisease_rfmid.json` | multidisease / RFMiD | strict_cross_domain |
| `configs/source_isolated/strict_cross_domain/odir_patient_multilabel_odir_5k.json` | odir_patient_multilabel / ODIR-5K | strict_cross_domain |
| `configs/source_isolated/within_source/image_quality_fives_fives.json` | image_quality / FIVES | within_source |
| `configs/source_isolated/within_source/landmark_localization_idrid.json` | landmark_localization / IDRiD | within_source |
| `configs/source_isolated/within_source/vessel_segmentation_fives.json` | vessel_segmentation / FIVES | within_source |

## 数据和任务约定

- 默认 `strict_cross_domain` 将IDRiD、FIVES和RFMiD2留作外测。血管分割和定位使用 `within_source` 才有训练样本；不能在来源内训练后再把默认协议外测称为未接触数据。
- 多标签损失仅使用已知标签掩码；未知标签不能当阴性。ODIR采用患者双眼输入，共享主干融合后预测患者标签，不把双眼拆成两个单眼标签。
- 分类配置的类别权重仅由训练标签计算并固定保存：多类使用平方根逆频率，多标签使用阴阳性数量比的平方根并限制最大值10；没有使用验证或测试标签计算权重。多标签阈值默认0.5，若调整阈值只能使用验证集。
- EyeQ三级质量和DDR不可判读二分类分别训练。FIVES三个质量标签在默认协议没有训练监督，不能直接用未训练头进行外测。
- 训练用分割掩膜是 `prepared/assets/masks` 下的单通道0/255 PNG。原始REFUGE掩膜包含0/128/255，不能直接当作二值目标。PAPILA两专家保持分开，专家选择在配置中固定。
- 分割的几何变换同步作用于图像与掩膜。DDR病灶按原分辨率随机512局部块训练，验证和测试将整图缩放至512；当前Dice/IoU不是原分辨率滑窗或病灶级指标。
- 检测配置明确使用 `source_xyxy_no_offset`，把原始数值作为连续xyxy，不擅自偏移；这是一种记录在案的实验约定，不声称来源已说明官方VOC坐标原点。检测评估为缩放图像上的COCO式AP。定位保留原像素数值并按图像尺寸归一化，评估同时记录相应距离定义。
- 类别不平衡、缺少患者身份、公开镜像预处理及预训练数据潜在重叠均影响结果解释。保留配置、来源和版本记录，不把重复文件数当独立病例数。
- EyeQ等来源的云端处理及参赛使用范围在本地来源说明中仍有待确认项目。数据可下载不代表任意再分发；上传与使用时保留各来源说明。

## 本地验证范围

本机可使用现有CPU版PyTorch进行 `--device cpu --smoke-test --workers 0` 调试。双卡DDP、fp16和显存占用必须在云服务器验证。本仓库不把CPU小样本验证表述为V100已完成训练。

已完成13项回归测试、真实图像分类/分割/定位的小批量前后向、FasterRCNN检测前后向和COCO式AP评估、两进程Gloo的三轮训练及冻结阶段切换；分段续训与连续训练的最终模型参数逐项相等。实际DINOv2-S主干也完成了随机初始化下的前后向验证。这些检查不代表预训练权重已下载、模型精度已达标或Linux双V100已实测。

可在环境安装后执行 `python -m unittest discover -s tests -v`。验证记录保存在 `TRAINING_VERIFICATION.json`。当前提供DINOv2-S基线和DINOv3-L对照；RETFound的原版MAE权重加载尚未实现，不能把普通ViT或任意本地权重标记为RETFound实验。
