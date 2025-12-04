"""
Text2MIDI 带思维链对齐版本

思维链对齐的核心思路：
    1. 文本描述 → 预测音乐属性（流派、调性、速度等）
    2. 文本 + 预测的属性 → 生成 MIDI

这样模型显式地学习了：
    - "欢快的钢琴曲" → 预测: 流派=古典, 速度=快, 调性=大调
    - 这些属性 → 影响 MIDI 生成

训练目标：
    1. 属性预测损失：让模型学会从文本预测音乐属性
    2. MIDI 生成损失：让模型学会根据文本+属性生成 MIDI
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import json

# 添加 MuseFormer 路径
MUSEFORMER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'muzic', 'museformer')
sys.path.insert(0, MUSEFORMER_PATH)


# ============================================
# 音乐属性定义
# ============================================

MUSIC_ATTRIBUTES = {
    'genre': ['classical', 'electronic', 'jazz', 'pop', 'rock', 'other'],
    'mood': ['happy', 'sad', 'calm', 'energetic', 'dramatic', 'other'],
    'tempo': ['slow', 'medium', 'fast'],  # <90, 90-140, >140 BPM
    'key_mode': ['major', 'minor'],
    'time_signature': ['4/4', '3/4', '6/8', 'other'],
}

NUM_ATTRIBUTES = sum(len(v) for v in MUSIC_ATTRIBUTES.values())


class AttributePredictor(nn.Module):
    """
    音乐属性预测器

    从文本特征预测音乐属性（多标签分类）
    """
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()

        self.attribute_heads = nn.ModuleDict()

        for attr_name, attr_values in MUSIC_ATTRIBUTES.items():
            self.attribute_heads[attr_name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, len(attr_values))
            )

    def forward(self, text_features):
        """
        Args:
            text_features: (batch, input_dim) - 池化后的文本特征

        Returns:
            predictions: dict of {attr_name: (batch, num_classes)}
        """
        predictions = {}
        for attr_name, head in self.attribute_heads.items():
            predictions[attr_name] = head(text_features)
        return predictions


class AttributeEncoder(nn.Module):
    """
    属性编码器

    将预测的属性编码为向量，用于条件化 MIDI 生成
    """
    def __init__(self, output_dim=512):
        super().__init__()

        self.embeddings = nn.ModuleDict()

        for attr_name, attr_values in MUSIC_ATTRIBUTES.items():
            self.embeddings[attr_name] = nn.Embedding(len(attr_values), output_dim // len(MUSIC_ATTRIBUTES))

        # 合并所有属性的投影
        total_embed_dim = (output_dim // len(MUSIC_ATTRIBUTES)) * len(MUSIC_ATTRIBUTES)
        self.projection = nn.Linear(total_embed_dim, output_dim)

    def forward(self, attribute_indices):
        """
        Args:
            attribute_indices: dict of {attr_name: (batch,) indices}

        Returns:
            attr_embedding: (batch, output_dim)
        """
        embeddings = []
        for attr_name in MUSIC_ATTRIBUTES.keys():
            if attr_name in attribute_indices:
                emb = self.embeddings[attr_name](attribute_indices[attr_name])
                embeddings.append(emb)

        concat = torch.cat(embeddings, dim=-1)
        return self.projection(concat)


class TextConditionEncoderWithCoT(nn.Module):
    """
    带思维链的文本条件编码器

    流程:
        1. 文本 → Qwen → 文本特征
        2. 文本特征 → 属性预测（思维链）
        3. 文本特征 + 属性编码 → 条件向量
    """
    def __init__(
        self,
        text_dim=1024,
        output_dim=512,
        hidden_dim=1024,
    ):
        super().__init__()

        self.output_dim = output_dim

        # 属性预测器（思维链）
        self.attribute_predictor = AttributePredictor(text_dim, hidden_dim)

        # 属性编码器
        self.attribute_encoder = AttributeEncoder(output_dim)

        # 文本投影
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # 融合层：文本特征 + 属性编码 → 最终条件
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_features, return_attributes=False):
        """
        Args:
            text_features: (batch, text_dim) - 池化后的文本特征
            return_attributes: 是否返回属性预测

        Returns:
            condition: (batch, output_dim) - 条件向量
            attr_logits: dict of attribute predictions (如果 return_attributes=True)
        """
        # 1. 预测属性（思维链）
        attr_logits = self.attribute_predictor(text_features)

        # 2. 获取预测的属性索引
        attr_indices = {}
        for attr_name, logits in attr_logits.items():
            attr_indices[attr_name] = logits.argmax(dim=-1)

        # 3. 编码属性
        attr_embedding = self.attribute_encoder(attr_indices)

        # 4. 投影文本特征
        text_embedding = self.text_projection(text_features)

        # 5. 融合
        combined = torch.cat([text_embedding, attr_embedding], dim=-1)
        condition = self.fusion(combined)

        if return_attributes:
            return condition, attr_logits
        return condition


class Text2MIDIWithCoT(nn.Module):
    """
    Text2MIDI 带思维链对齐版本

    训练时的损失：
        1. 属性预测损失（CE Loss）
        2. MIDI 生成损失（CE Loss）

    推理时的流程：
        1. 输入文本 → 预测属性（可以打印出来看）
        2. 文本 + 属性 → 生成 MIDI
    """
    def __init__(
        self,
        qwen_model_path,
        museformer_checkpoint_path,
        museformer_data_dir,
        hidden_dim=1024,
        freeze_encoder=True,
        freeze_decoder=True,
        attribute_loss_weight=0.1,  # 属性预测损失的权重
        device='cpu'
    ):
        super().__init__()

        self.attribute_loss_weight = attribute_loss_weight

        # 1. 文本编码器 (Qwen)
        print("加载 Qwen Encoder...")
        self.text_encoder = AutoModel.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        self.text_dim = self.text_encoder.config.hidden_size

        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("  Qwen Encoder 已冻结")

        # 2. MuseFormer
        print("加载 MuseFormer...")
        self.museformer, self.museformer_args, self.dictionary = self._load_museformer(
            museformer_checkpoint_path, museformer_data_dir
        )
        self.museformer_dim = self.museformer_args.attention_embed_dim

        if freeze_decoder:
            for param in self.museformer.parameters():
                param.requires_grad = False
            print("  MuseFormer Decoder 已冻结")

        # 3. 带思维链的条件编码器
        print("创建 TextConditionEncoder with CoT...")
        self.condition_encoder = TextConditionEncoderWithCoT(
            text_dim=self.text_dim,
            output_dim=self.museformer_dim,
            hidden_dim=hidden_dim
        )

        # 特殊 token
        self.bos_idx = self.dictionary.bos()
        self.eos_idx = self.dictionary.eos()
        self.pad_idx = self.dictionary.pad()

        print(f"\n模型配置:")
        print(f"  Text dim: {self.text_dim}")
        print(f"  MuseFormer dim: {self.museformer_dim}")
        print(f"  属性预测: {list(MUSIC_ATTRIBUTES.keys())}")

    def _load_museformer(self, checkpoint_path, data_dir):
        """加载 MuseFormer"""
        from fairseq.data import Dictionary
        from museformer.museformer_lm_task import MuseformerLanguageModelingTask

        dict_path = os.path.join(data_dir, 'dict.txt')
        dictionary = Dictionary.load(dict_path)

        state = torch.load(checkpoint_path, map_location='cpu')
        model_args = state['args']
        model_args.data = data_dir

        task = MuseformerLanguageModelingTask.setup_task(model_args)
        model = task.build_model(model_args)
        model.load_state_dict(state['model'], strict=True)

        return model, model_args, dictionary

    def encode_text(self, input_ids, attention_mask, return_attributes=False):
        """编码文本"""
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return self.condition_encoder(pooled, return_attributes=return_attributes)

    def forward(self, input_ids, attention_mask, midi_tokens, attribute_labels=None):
        """
        训练时的前向传播

        Args:
            input_ids: 文本 token ids
            attention_mask: 文本 attention mask
            midi_tokens: MIDI token ids
            attribute_labels: dict of ground truth attribute labels (可选)

        Returns:
            logits: MIDI token logits
            attr_logits: 属性预测 logits
            attr_loss: 属性预测损失 (如果提供了 labels)
        """
        # 编码文本并获取属性预测
        condition, attr_logits = self.encode_text(input_ids, attention_mask, return_attributes=True)

        # MuseFormer 前向传播
        logits, _ = self.museformer(midi_tokens)

        # 计算属性预测损失
        attr_loss = None
        if attribute_labels is not None:
            attr_loss = 0
            for attr_name, labels in attribute_labels.items():
                if attr_name in attr_logits:
                    attr_loss += F.cross_entropy(attr_logits[attr_name], labels)
            attr_loss = attr_loss / len(attribute_labels)

        return logits, attr_logits, attr_loss

    @torch.no_grad()
    def generate(self, text, max_length=1024, temperature=1.0, top_k=8, device='cpu', verbose=True):
        """生成 MIDI，并打印思维链"""
        self.eval()

        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        condition, attr_logits = self.encode_text(input_ids, attention_mask, return_attributes=True)

        # 打印思维链（预测的属性）
        if verbose:
            print("\n思维链（预测的音乐属性）:")
            for attr_name, logits in attr_logits.items():
                probs = F.softmax(logits[0], dim=-1)
                pred_idx = probs.argmax().item()
                pred_value = MUSIC_ATTRIBUTES[attr_name][pred_idx]
                confidence = probs[pred_idx].item()
                print(f"  {attr_name}: {pred_value} (置信度: {confidence:.2%})")

        # 生成
        generated = torch.LongTensor([[self.bos_idx]]).to(device)

        for step in range(max_length):
            logits, _ = self.museformer(generated)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == self.eos_idx:
                break

        token_list = generated[0].tolist()
        token_strings = [
            self.dictionary[t] for t in token_list
            if t not in [self.bos_idx, self.eos_idx, self.pad_idx]
        ]

        return token_list, token_strings, attr_logits


def extract_attribute_labels(item):
    """
    从 MidiCaps 数据项中提取属性标签

    Args:
        item: MidiCaps 数据项

    Returns:
        labels: dict of attribute labels
    """
    labels = {}

    # 流派
    genre = item.get('genre', [])
    if genre:
        main_genre = genre[0].lower()
        for idx, g in enumerate(MUSIC_ATTRIBUTES['genre']):
            if g in main_genre or main_genre in g:
                labels['genre'] = idx
                break
        if 'genre' not in labels:
            labels['genre'] = len(MUSIC_ATTRIBUTES['genre']) - 1  # other

    # 情绪
    mood = item.get('mood', [])
    if mood:
        main_mood = mood[0].lower()
        mood_map = {
            'happy': 0, 'cheerful': 0, 'joyful': 0,
            'sad': 1, 'melancholic': 1,
            'calm': 2, 'peaceful': 2, 'relaxing': 2,
            'energetic': 3, 'upbeat': 3,
            'dramatic': 4, 'epic': 4,
        }
        labels['mood'] = mood_map.get(main_mood, len(MUSIC_ATTRIBUTES['mood']) - 1)

    # 速度
    tempo = item.get('tempo', 120)
    if tempo < 90:
        labels['tempo'] = 0  # slow
    elif tempo < 140:
        labels['tempo'] = 1  # medium
    else:
        labels['tempo'] = 2  # fast

    # 调性
    key = item.get('key', '')
    if 'major' in key.lower():
        labels['key_mode'] = 0
    elif 'minor' in key.lower():
        labels['key_mode'] = 1
    else:
        labels['key_mode'] = 0  # default to major

    # 拍号
    time_sig = item.get('time_signature', '4/4')
    time_sig_map = {'4/4': 0, '3/4': 1, '6/8': 2}
    labels['time_signature'] = time_sig_map.get(time_sig, 3)

    return labels


# ============================================
# 配置
# ============================================

class ModelConfigWithCoT:
    """模型配置"""
    qwen_model_path = "/Users/freeninglu/Desktop/MuseFormer/Qwen3-Embedding-0.6B"
    museformer_checkpoint = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
    museformer_data_dir = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/data-bin/lmd6remi"

    hidden_dim = 1024
    attribute_loss_weight = 0.1

    freeze_encoder = True
    freeze_decoder = True


if __name__ == '__main__':
    print("=" * 60)
    print("Text2MIDI 带思维链对齐版本测试")
    print("=" * 60)

    # 测试属性提取
    test_item = {
        'genre': ['classical', 'orchestral'],
        'mood': ['dramatic', 'epic'],
        'tempo': 100,
        'key': 'C major',
        'time_signature': '4/4',
    }

    labels = extract_attribute_labels(test_item)
    print("\n测试属性提取:")
    print(f"  输入: {test_item}")
    print(f"  输出: {labels}")

    print("\n音乐属性定义:")
    for attr_name, attr_values in MUSIC_ATTRIBUTES.items():
        print(f"  {attr_name}: {attr_values}")
