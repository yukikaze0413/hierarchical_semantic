# 层级语义瓶颈 EEG-to-Text 方案设计与代码文档

项目英文名：

```text
Hierarchical Semantic Bottleneck EEG-to-Text
```

## 1. 方案核心想法

本方案将 EEG-to-Text 从“直接生成完整句子”或“预测扁平关键词”重构为一个**层级语义解码问题**。

核心流程是：

```text
EEG -> 粗粒度语义类别 -> 中层语义概念 -> 细粒度关键词 -> 句子重建
```

例如：

```text
EEG 片段
-> object
-> vehicle
-> ambulance
```

对应中文理解：

```text
脑电信号
-> 物体
-> 交通工具
-> 救护车
```

核心假设是：

```text
非侵入式 EEG 对高层语义信息的编码比对精确词语的编码更稳定。
```

因此，与其强迫 EEG 直接预测完整句子或精确单词，不如先恢复更可靠的高层语义锚点，再逐步细化到具体词，最后由 LLM 在语义约束下重建句子。

本方案结合三类近期思路：

```text
1. Brain-CLIPLM 的 semantic bottleneck / semantic anchor 思想；
2. Hierarchic-EEG2Text 的层级语义解码思想；
3. RAG + LLM 的受控句子重建思想。
```

## 2. 研究动机

现有 EEG-to-text 方法常见流程是：

```text
EEG -> keywords -> LLM sentence generation
```

也就是：

```text
脑电信号 -> 关键词 -> 大语言模型生成句子
```

这种 flat keyword 方法有三个问题。

第一，细粒度词语很难直接从 EEG 中稳定解码。

例如：

```text
ambulance, bus, car, truck
```

它们都属于 vehicle，EEG 可能能比较稳定地区分“交通工具”，但很难稳定区分到底是哪一种交通工具。

第二，传统 flat accuracy 把所有错误都当成同等错误。

例如真实词是：

```text
ambulance
```

预测为：

```text
bus
```

虽然 fine keyword 错了，但仍然保留了 vehicle 这一层语义。  
如果预测为：

```text
sadness
```

那才是语义完全跑偏。

传统 flat keyword accuracy 无法区分这两类错误。

第三，LLM 容易被错误的细粒度关键词误导。

如果 EEG 模型其实只确定是：

```text
vehicle
```

但错误输出了：

```text
ambulance
```

LLM 可能会强行围绕 ambulance 生成句子，导致重建偏离真实语义。

因此，本方案提出：

```text
不要只向 LLM 输入 flat keywords，
而是输入带层级结构和置信度的 semantic anchors。
```

例如：

```json
{
  "coarse": "object",
  "mid": "vehicle",
  "fine_candidates": [
    {"keyword": "ambulance", "confidence": 0.41},
    {"keyword": "bus", "confidence": 0.29},
    {"keyword": "car", "confidence": 0.18}
  ]
}
```

这样 LLM 可以知道：

```text
模型比较确定这是 vehicle，但不完全确定具体是哪种 vehicle。
```

## 3. 完整端到端管线

整体流程：

```text
原始 EEG + 句子文本标签
        |
        v
EEG 预处理
        |
        v
词级 EEG 对齐
        |
        v
内容词提取
        |
        v
层级语义标签构建
        |
        v
EEG-Text 对比学习训练
        |
        v
粗到细语义锚点解码
        |
        v
层级感知 RAG 检索
        |
        v
LLM 句子重建
        |
        v
评估与消融实验
```

更简洁地说：

```text
Raw EEG
-> Preprocessing
-> EEG Encoder
-> Hierarchical Semantic Anchors
-> Hierarchy-aware RAG
-> LLM Reconstruction
-> Reconstructed Sentence
```

## 4. 数据设计

### 4.1 推荐数据集

主数据集：

```text
ZuCo
```

用途：

```text
阅读 EEG -> 语义锚点 -> 句子重建
```

ZuCo 的优势是它包含自然阅读过程中的 EEG 和文本对齐信息，适合做 EEG-to-text 或 EEG-to-semantics。

可选辅助数据集：

```text
PEERS
```

用途：

```text
大规模层级物体 / 概念解码验证
```

建议实验设计：

```text
Experiment 1: 在 PEERS 上验证层级语义解码是否有效；
Experiment 2: 在 ZuCo 上验证层级语义锚点是否提升句子重建。
```

如果先做最小可行版本，只用 ZuCo 也可以。

### 4.2 词级样本格式

每个 word-level EEG 样本可以整理成：

```json
{
  "subject_id": "S01",
  "sentence_id": "sent_0001",
  "word_id": 5,
  "word": "ambulance",
  "lemma": "ambulance",
  "sentence": "The ambulance stopped near the university hospital.",
  "eeg": "path/to/eeg_tensor.npy",
  "coarse": "object",
  "mid": "vehicle",
  "fine": "ambulance"
}
```

### 4.3 句子级样本格式

每个 sentence-level 样本可以整理成：

```json
{
  "sentence_id": "sent_0001",
  "sentence": "The ambulance stopped near the university hospital.",
  "anchors": [
    ["object", "vehicle", "ambulance"],
    ["action", "motion", "stop"],
    ["place", "institution", "university"],
    ["place", "medical_facility", "hospital"]
  ]
}
```

## 5. EEG 预处理

推荐预处理步骤：

```text
1. Bandpass filter: 0.5-40 Hz
2. Notch filter: 50 Hz 或 60 Hz
3. Artifact removal: ICA 或 ASR
4. Downsample: 250 Hz
5. Epoching: 以 word fixation onset 或 word onset 为中心，截取 -200 ms 到 800 ms
6. Baseline correction: 使用 -200 ms 到 0 ms
7. Bad channel interpolation
8. Z-score normalization: 只使用训练集统计量
```

可以使用两类 EEG 表示。

表示 A：时频张量

```text
48 channels x 5 frequency bands x 10 time windows
```

表示 B：原始时间序列

```text
48 channels x 250 time points
```

建议第一版使用表示 A，因为它更接近 Brain-CLIPLM，方便对比。

## 6. 文本处理与层级语义标签构建

### 6.1 内容词提取

对每个句子：

```text
1. 分词；
2. POS tagging；
3. lemmatization；
4. 保留 nouns、verbs、adjectives 和部分 adverbs；
5. 去掉 function words、数字、月份、星期和极低频词。
```

例子：

```text
Sentence:
The ambulance stopped near the university hospital.

Content words:
ambulance, stopped, university, hospital
```

### 6.2 关键词词表构建

从训练集构建 controlled keyword vocabulary。

流程：

```text
1. 统计内容词频率；
2. 保留频率超过阈值的词；
3. 用 lemma 合并词形变化；
4. 删除噪声词和语义不清晰的词；
5. 使用 GloVe 或 SBERT embedding 去除语义冗余；
6. 选出 V 个关键词。
```

推荐词表规模：

```text
V = 100    原型实验
V = 300    主实验
V = 500+   扩展实验
```

### 6.3 层级标签映射

对每个 fine keyword 分配三层语义标签：

```text
coarse category
mid-level concept
fine keyword
```

例如：

```text
ambulance -> object -> vehicle -> ambulance
doctor -> person -> medical_professional -> doctor
university -> place -> institution -> university
run -> action -> movement -> run
painful -> attribute -> sensation -> painful
```

可以整理成 taxonomy 表：

```csv
keyword,coarse,mid,fine
ambulance,object,vehicle,ambulance
doctor,person,medical_professional,doctor
university,place,institution,university
run,action,movement,run
painful,attribute,sensation,painful
```

推荐构建方式：

```text
1. WordNet 作为主要层级来源；
2. ConceptNet 作为 commonsense fallback；
3. LLM 辅助处理未覆盖或有歧义的词；
4. 人工抽查和修正最终 taxonomy。
```

## 7. 模型结构

### 7.1 总体结构

模型包含 EEG 分支和文本分支。

```text
EEG branch:
EEG tensor -> EEG encoder -> EEG embedding

Text branch:
coarse label -> Text encoder -> coarse embedding
mid label    -> Text encoder -> mid embedding
fine keyword -> Text encoder -> fine embedding
```

训练目标：

```text
让 EEG embedding 同时接近对应的 coarse、mid、fine 文本 embedding。
```

### 7.2 EEG Encoder

可选编码器：

```text
1. Deep4-style CNN
2. EEGNet
3. Conformer
4. Temporal CNN + Transformer
```

推荐原型：

```text
Deep4-style CNN
```

输入：

```text
[batch_size, channels, frequency_bands, time_windows]
```

输出：

```text
[batch_size, embedding_dim]
```

建议 embedding 维度：

```text
128 或 256
```

### 7.3 Text Encoder

可选文本编码器：

```text
1. Sentence-BERT
2. BERT-base
3. ClinicalBERT，适合临床 EEG report 变体
```

推荐原型：

```text
Frozen Sentence-BERT + trainable projection layer
```

## 8. 训练目标

总损失：

```text
L_total =
  lambda_c * L_coarse
+ lambda_m * L_mid
+ lambda_f * L_fine
+ lambda_h * L_hierarchy
```

推荐权重：

```text
lambda_c = 0.5
lambda_m = 0.7
lambda_f = 1.0
lambda_h = 0.3
```

### 8.1 对比学习损失

每一层使用 symmetric InfoNCE：

```text
L_level = InfoNCE(EEG_embedding, text_embedding_at_level)
```

作用：

```text
拉近匹配的 EEG-text pair；
推远不匹配的 EEG-text pair。
```

例如：

```text
EEG(ambulance) 应该接近：
object embedding
vehicle embedding
ambulance embedding

EEG(ambulance) 应该远离：
emotion embedding
food embedding
doctor embedding
```

### 8.2 层级一致性损失

层级一致性损失用于惩罚不合法路径。

例如：

```text
ambulance 必须属于 vehicle；
vehicle 必须属于 object。
```

不应该出现：

```text
object -> emotion -> ambulance
```

简单实现：

```text
L_hierarchy = CE(P(mid | EEG), true_mid) + CE(P(coarse | EEG), true_coarse)
```

也可以用图距离：

```text
penalty = tree_distance(predicted_node, ground_truth_node)
```

### 8.3 课程学习训练策略

推荐使用 coarse-to-fine curriculum。

训练日程：

```text
Epoch 1-20:
  L = L_coarse + L_mid

Epoch 21-50:
  L = L_coarse + L_mid + L_fine

Epoch 51+:
  L = L_total
```

直觉：

```text
先让模型学会稳定的高层语义结构，
再逐渐学习更难的细粒度关键词。
```

## 9. 粗到细推理流程

输入一个 EEG segment：

```text
EEG segment -> EEG encoder -> EEG embedding
```

第一步，预测 coarse category：

```text
在所有 coarse categories 中计算 cosine similarity。
```

例如：

```text
object: 0.78
place: 0.21
emotion: 0.08
```

第二步，在选中的 coarse 下预测 mid concept：

```text
object 的候选 mid:
vehicle, tool, building, furniture
```

例如：

```text
vehicle: 0.66
tool: 0.14
building: 0.10
```

第三步，在选中的 mid 下预测 fine keyword：

```text
vehicle 的候选 fine:
ambulance, bus, car, train
```

例如：

```text
ambulance: 0.41
bus: 0.29
car: 0.18
```

输出：

```json
{
  "coarse": {
    "label": "object",
    "score": 0.78
  },
  "mid": {
    "label": "vehicle",
    "score": 0.66
  },
  "fine_candidates": [
    {"keyword": "ambulance", "score": 0.41},
    {"keyword": "bus", "score": 0.29},
    {"keyword": "car", "score": 0.18}
  ]
}
```

句子级别聚合：

```text
word segment 1 -> object > vehicle > ambulance
word segment 2 -> action > motion > stop
word segment 3 -> place > institution > university
word segment 4 -> place > medical_facility > hospital
```

句子级 semantic anchors：

```text
[
  object > vehicle > ambulance,
  action > motion > stop,
  place > institution > university,
  place > medical_facility > hospital
]
```

如果 fine-level confidence 低，则回退到 mid-level：

```text
object > vehicle > [fine uncertain]
```

而不是强行输出一个可能错误的 fine keyword。

## 10. 层级感知 RAG 检索

为训练集中的句子建立检索索引。

每个索引条目包含：

```json
{
  "sentence": "The ambulance stopped near the university hospital.",
  "flat_keywords": ["ambulance", "stop", "university", "hospital"],
  "coarse_labels": ["object", "action", "place"],
  "mid_labels": ["vehicle", "motion", "institution", "medical_facility"],
  "hierarchical_paths": [
    "object > vehicle > ambulance",
    "action > motion > stop",
    "place > institution > university",
    "place > medical_facility > hospital"
  ],
  "sbert_embedding": "..."
}
```

检索分数：

```text
score =
  alpha * fine_keyword_overlap
+ beta  * mid_concept_overlap
+ gamma * coarse_category_overlap
+ delta * sentence_embedding_similarity
```

推荐权重：

```text
alpha = 0.4
beta = 0.3
gamma = 0.1
delta = 0.2
```

这样做的意义：

```text
即使 fine keyword 不准确，mid-level concept 仍然能帮助检索到语义相近的参考句。
```

## 11. LLM 句子重建

LLM 的输入不再只是 flat keywords，而是：

```text
层级语义锚点 + 置信度 + RAG 检索句
```

Prompt 模板：

```text
You are reconstructing a sentence from EEG-derived hierarchical semantic anchors.

The EEG decoder is more reliable at coarse and mid semantic levels than at exact word level.
Use fine keywords when confidence is high.
When fine keywords are uncertain, preserve the mid-level meaning instead of forcing exact words.

Hierarchical anchors:
1. object > vehicle > ambulance, confidence: 0.41
2. action > motion > stop, confidence: 0.46
3. place > institution > university, confidence: 0.58
4. place > medical_facility > hospital, confidence: 0.52

Retrieved reference sentences:
1. ...
2. ...
3. ...

Generate one fluent English sentence that is semantically consistent with the anchors.
```

预期输出：

```text
The ambulance stopped near the university hospital.
```

为了方便评估，也可以要求 LLM 输出结构化结果：

```json
{
  "entities": ["ambulance", "university hospital"],
  "action": "stopped near",
  "sentence": "The ambulance stopped near the university hospital."
}
```

## 12. 评估指标

### 12.1 层级解码指标

```text
Coarse Top-1 accuracy
Coarse Top-5 accuracy
Mid Top-1 accuracy
Mid Top-5 accuracy
Fine Top-1 accuracy
Fine Top-5 accuracy
MRR
Hierarchical distance
Lowest common ancestor depth
```

其中 hierarchical distance 用来衡量语义错误严重程度。

例子：

```text
Ground truth:
object > vehicle > ambulance

Prediction A:
object > vehicle > bus

Prediction B:
emotion > sadness > grief
```

Prediction A 虽然 fine keyword 错了，但语义接近；Prediction B 则完全跑偏。  
hierarchical distance 应该能反映这种差异。

### 12.2 句子重建指标

```text
Sentence retrieval Top-5 / Top-10 / Top-25
SBERT similarity
BERTScore
ROUGE-L
Keyword coverage
Concept coverage
Hierarchy coverage
```

其中 hierarchy coverage 是本方案新增重点指标：

```text
生成句子是否保留了解码出的 coarse / mid / fine 语义。
```

### 12.3 鲁棒性指标

```text
Noise EEG performance
Shuffled EEG-label control
Random hierarchy control
Frequency-matched random anchor control
Cross-subject generalization
Leave-one-subject-out evaluation
```

这些控制实验用于证明：

```text
性能提升不是 LLM 自己猜出来的，
也不是随机层级结构带来的假提升。
```

## 13. Baseline 与消融实验

### 13.1 Baseline

推荐 baseline：

```text
1. Flat keyword retrieval
   EEG -> fine keyword -> LLM reconstruction

2. Fine-only contrastive model
   EEG 只和 fine keyword embedding 对齐

3. Coarse-only model
   EEG 只和 coarse label embedding 对齐

4. Sentence embedding retrieval
   EEG -> sentence embedding -> nearest candidate sentence

5. Closed-set classifier
   EEG -> softmax keyword classification

6. Random hierarchy
   使用随机 coarse / mid 映射

7. Oracle hierarchy
   使用真实层级锚点重建句子，估计上限
```

### 13.2 关键消融

```text
1. 去掉 hierarchy consistency loss；
2. 去掉 curriculum training；
3. 将 WordNet hierarchy 替换为 random hierarchy；
4. LLM prompt 使用 flat keywords，而不是 hierarchical anchors；
5. 去掉 hierarchy-aware RAG；
6. 测试不同词表规模：100、300、500。
```

## 14. 建议代码目录结构

```text
project_root/
  configs/
    zuco_hierarchical.yaml
    peers_hierarchical.yaml

  data/
    raw/
    processed/
    taxonomy/
      keyword_taxonomy.csv
      hierarchy_edges.csv

  src/
    preprocessing/
      eeg_preprocess.py
      text_preprocess.py
      build_epochs.py

    taxonomy/
      build_wordnet_taxonomy.py
      build_conceptnet_fallback.py
      validate_taxonomy.py

    datasets/
      zuco_dataset.py
      peers_dataset.py
      collate.py

    models/
      eeg_encoder.py
      text_encoder.py
      hierarchical_decoder.py
      losses.py

    training/
      train_hierarchical.py
      evaluate_hierarchy.py
      checkpointing.py

    inference/
      decode_anchors.py
      aggregate_sentence_anchors.py
      retrieve_rag_examples.py
      reconstruct_with_llm.py

    evaluation/
      metrics_hierarchy.py
      metrics_generation.py
      robustness_controls.py
      ablations.py

  scripts/
    01_preprocess_zuco.py
    02_build_taxonomy.py
    03_train_model.py
    04_decode_test_set.py
    05_reconstruct_sentences.py
    06_evaluate.py

  outputs/
    checkpoints/
    decoded_anchors/
    reconstructed_sentences/
    reports/
```

## 15. 核心伪代码

### 15.1 训练伪代码

```python
for batch in train_loader:
    eeg = batch["eeg"]
    coarse_text = batch["coarse_text"]
    mid_text = batch["mid_text"]
    fine_text = batch["fine_text"]

    eeg_emb = eeg_encoder(eeg)
    coarse_emb = text_encoder(coarse_text)
    mid_emb = text_encoder(mid_text)
    fine_emb = text_encoder(fine_text)

    loss_coarse = info_nce(eeg_emb, coarse_emb)
    loss_mid = info_nce(eeg_emb, mid_emb)
    loss_fine = info_nce(eeg_emb, fine_emb)

    loss_hierarchy = hierarchy_consistency_loss(
        eeg_emb=eeg_emb,
        true_coarse=batch["coarse_id"],
        true_mid=batch["mid_id"],
        true_fine=batch["fine_id"],
        hierarchy=taxonomy
    )

    loss = (
        lambda_c * loss_coarse
        + lambda_m * loss_mid
        + lambda_f * loss_fine
        + lambda_h * loss_hierarchy
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 15.2 粗到细解码伪代码

```python
def decode_anchor(eeg_segment, model, taxonomy, top_k=5):
    eeg_emb = model.encode_eeg(eeg_segment)

    coarse_scores = cosine_scores(eeg_emb, taxonomy.coarse_embeddings)
    top_coarse = select_topk(coarse_scores, k=top_k)

    anchor_paths = []
    for coarse in top_coarse:
        mid_candidates = taxonomy.children_of(coarse.label)
        mid_scores = cosine_scores(eeg_emb, mid_candidates.embeddings)
        top_mid = select_topk(mid_scores, k=top_k)

        for mid in top_mid:
            fine_candidates = taxonomy.children_of(mid.label)
            fine_scores = cosine_scores(eeg_emb, fine_candidates.embeddings)
            top_fine = select_topk(fine_scores, k=top_k)

            for fine in top_fine:
                anchor_paths.append({
                    "coarse": coarse.label,
                    "mid": mid.label,
                    "fine": fine.label,
                    "score": coarse.score * mid.score * fine.score
                })

    return sorted(anchor_paths, key=lambda x: x["score"], reverse=True)
```

### 15.3 句子级锚点聚合伪代码

```python
def aggregate_sentence_anchors(word_anchor_predictions, max_anchors=5):
    candidates = []

    for pred in word_anchor_predictions:
        best_path = pred[0]

        if best_path["score"] < 0.3:
            best_path["fine"] = None

        candidates.append(best_path)

    candidates = merge_duplicate_paths(candidates)
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    return candidates[:max_anchors]
```

### 15.4 层级感知 RAG 检索伪代码

```python
def retrieve_examples(anchor_paths, index, top_k=5):
    query_fine = set(p["fine"] for p in anchor_paths if p["fine"])
    query_mid = set(p["mid"] for p in anchor_paths)
    query_coarse = set(p["coarse"] for p in anchor_paths)

    scored = []

    for item in index:
        fine_overlap = jaccard(query_fine, item.fine_keywords)
        mid_overlap = jaccard(query_mid, item.mid_labels)
        coarse_overlap = jaccard(query_coarse, item.coarse_labels)
        semantic_sim = sbert_similarity(anchor_paths, item.sbert_embedding)

        score = (
            0.4 * fine_overlap
            + 0.3 * mid_overlap
            + 0.1 * coarse_overlap
            + 0.2 * semantic_sim
        )

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scored[:top_k]]
```

## 16. 最小可行实验版本

建议第一版这样做：

```text
Dataset: ZuCo
Vocabulary size: 100 keywords
Hierarchy: WordNet + manual cleanup
EEG input: 48 x 5 x 10 tensor
EEG encoder: Deep4 CNN
Text encoder: frozen SBERT
Training: coarse / mid / fine InfoNCE
Inference: coarse-to-fine retrieval
Reconstruction: LLM + top-5 hierarchy-aware RAG
```

最小 baseline：

```text
1. Flat keyword retrieval
2. Fine-only contrastive model
3. Random hierarchy
4. Oracle hierarchy
```

最小指标：

```text
1. Coarse / mid / fine Top-k accuracy
2. Hierarchical distance
3. Sentence Top-5 retrieval
4. SBERT similarity
5. Shuffled-label robustness control
```

## 17. 预期贡献

贡献一：

```text
提出 Hierarchical Semantic Bottleneck，用层级语义锚点替代扁平关键词作为 EEG-to-text 的中间表示。
```

贡献二：

```text
提出 coarse-to-fine EEG-text contrastive learning，在 coarse、mid、fine 三个语义层面同时对齐 EEG 与文本表示。
```

贡献三：

```text
提出 hierarchy-aware RAG + LLM reconstruction，使句子重建同时利用细粒度关键词和更可靠的中高层语义。
```

贡献四：

```text
提出层级感知评价指标，用 hierarchical distance 衡量语义错误严重程度，而不是把所有关键词错误都视为相同。
```

## 18. 论文需要证明的核心结论

主结论：

```text
对于非侵入式 EEG-to-text，层级语义锚点比扁平关键词是更合适的语义瓶颈。
```

需要通过实验证明：

```text
1. coarse 和 mid-level concepts 比 fine keywords 更容易从 EEG 中稳定解码；
2. coarse-to-fine decoding 比 flat fine keyword retrieval 更稳；
3. hierarchical anchors 能提升 LLM sentence reconstruction；
4. random hierarchy 不能带来同样提升；
5. 该方法在噪声、标签打乱、跨被试场景下更鲁棒。
```

## 19. 一句话总结

```text
本方案不是强迫低信噪比 EEG 直接识别精确词或完整句子，
而是让 EEG 先恢复更可靠的层级语义结构，
再用 RAG 和 LLM 将这些语义锚点重建为自然语言。
```

