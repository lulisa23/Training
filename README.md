# 电阻率机器学习辅助设计

本仓库提供一套面向 500 C 与 600 C 电阻率数据的轻量机器学习流程：

1. 把实验表整理成统一 CSV/XLSX。
2. 用图三中的物理/化学/工艺特征训练电阻率预测模型。
3. 对候选化合物批量预测电阻率。
4. 按“最小”或“最大”电阻率排序，辅助筛选值得实验验证的配方。

## 特征列

训练和预测脚本使用以下特征列，含义与图三一致：

| 类别 | 特征 |
| --- | --- |
| 尺寸因素 | `rA`, `rB`, `VA`, `VB` |
| 电化学因素 | `Octahed`, `EN_PA`, `EN_PB` |
| 键合参数 | `EA`, `EB`, `EAA`, `EAB`, `DA`, `DB`, `IPA`, `IPB`, `NMA`, `NMB` |
| 原子因素 | `A`, `B`, `MASS`, `NUMBER` |
| 物理因素 | `TmA`, `TmB` |
| 工艺因素 | `Ts`, `ts`, `th` |
| 可选温度特征 | `temperature_c` |

CSV 模板位于：

- `data/resistivity_measurements_template.csv`：训练数据模板，包含目标列 `rho`。
- `data/candidates_template.csv`：候选化合物模板，不包含目标列。

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

## 数据整理

推荐把 500 C 与 600 C 数据整理成长表：

```csv
compound,formula,A_site,B_site,temperature_c,rA,...,th,rho
sample_1,La0.8Sr0.2...,La/Sr,...,500,1.34,...,24,1.0e-6
sample_1,La0.8Sr0.2...,La/Sr,...,600,1.34,...,24,7.5e-7
```

注意：

- `rho` 必须是正数，且所有行单位一致。
- 电阻率常跨越多个数量级，脚本默认训练 `log10(rho)`。
- 如果某个特征暂时缺失，可留空；模型会进行中位数填补。
- 如果“最佳电阻率”代表低电阻率，用预测脚本的 `--direction min`；如果代表高电阻率，用 `--direction max`。

## 训练 500 C 和 600 C 独立模型

```bash
python3 -m resistivity_ml.train \
  --data data/resistivity_measurements.csv \
  --temperature 500 \
  --model-out models/rho_500c.joblib \
  --report-out reports/rho_500c_report.json

python3 -m resistivity_ml.train \
  --data data/resistivity_measurements.csv \
  --temperature 600 \
  --model-out models/rho_600c.joblib \
  --report-out reports/rho_600c_report.json
```

## 训练跨温度模型

如果希望模型同时学习温度影响：

```bash
python3 -m resistivity_ml.train \
  --data data/resistivity_measurements.csv \
  --include-temperature \
  --model-out models/rho_combined.joblib \
  --report-out reports/rho_combined_report.json
```

此时候选表也必须包含 `temperature_c`。

## 候选化合物预测与排序

```bash
python3 -m resistivity_ml.predict \
  --model models/rho_500c.joblib \
  --candidates data/candidates.csv \
  --direction min \
  --out reports/candidates_ranked_500c.csv \
  --top 20
```

## 一键完整脚本

如果你的文件就是 `features_500.xlsx` 和 `features_600.xlsx`，目标列是
`rho_clean`，可以直接运行：

```bash
python3 scripts/complete_resistivity_workflow.py \
  --data-dir "D:/Machine learning" \
  --file-500 features_500.xlsx \
  --file-600 features_600.xlsx \
  --target rho_clean
```

脚本会自动完成：

- 读取 500 C 与 600 C Excel 数据；
- 输出与 `rho_clean` 的相关系数排序；
- 保存相关性热力图；
- 用 `log10(rho_clean)` 训练 Ridge、RandomForest、ExtraTrees、GradientBoosting；
- 交叉验证比较模型效果；
- 保存最佳模型；
- 保存前 20 个重要特征；
- 如果提供候选表，还会预测并排序候选化合物。

候选预测示例：

```bash
python3 scripts/complete_resistivity_workflow.py \
  --data-dir "D:/Machine learning" \
  --file-500 features_500.xlsx \
  --file-600 features_600.xlsx \
  --candidate-500 candidates_500.xlsx \
  --candidate-600 candidates_600.xlsx \
  --target rho_clean \
  --direction min
```

输出列包括：

- `predicted_log10_rho`
- `predicted_rho`
- `rank`
- 对树模型还会给出 `prediction_std_log10` 和 `rho_factor_1std`，用于粗略判断模型不确定性。

## 建模策略

训练脚本会比较多个适合小样本表格数据的模型：

- Ridge 回归
- Random Forest
- Extra Trees
- Gradient Boosting

报告文件中会记录交叉验证误差、最佳模型和前 15 个重要特征。小样本时应重点关注：

1. `rmse_log10`：例如 0.3 约等于 2 倍数量级误差。
2. 重要特征是否符合材料机理直觉。
3. 候选预测的不确定性；不确定性高的样品适合作为探索性实验，而不是直接视为确定最优。
