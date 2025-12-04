"""
Text2MIDI 最终版本
实现文本条件化的 MIDI 生成

核心思路：
    将文本特征注入到 MuseFormer 的 token embedding 中
    这样 MuseFormer 在生成时能"感知"文本描述

实现方式：
    1. Qwen 编码文本 → (batch, text_len, 1024)
    2. 池化 → (batch, 1024)
    3. 投影为 bias 向量 → (batch, 512)
    4. 将 bias 加到每个 MIDI token 的 embedding 上
    5. MuseFormer 使用修改后的 embedding 生成

这是侵入性最小的方案，因为：
    - 不修改 MuseFormer 的注意力结构
    - 只在 embedding 层添加条件
    - MuseFormer 的音乐生成能力完全保留
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import math
import copy

# 添加 MuseFormer 路径
MUSEFORMER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'muzic', 'museformer')
sys.path.insert(0, MUSEFORMER_PATH)


class TextConditionEncoder(nn.Module):
    """
    文本条件编码器

    将文本特征转换为可以注入到 MuseFormer 的条件向量
    """
    def __init__(
        self,
        text_dim=1024,
        output_dim=512,
        hidden_dim=1024,
        num_heads=8,
    ):
        super().__init__()

        self.output_dim = output_dim

        # 方案1: 全局 bias（最简单）
        self.global_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # 方案2: 可学习的 scale 和 shift (FiLM)
        self.scale_projection = nn.Linear(text_dim, output_dim)
        self.shift_projection = nn.Linear(text_dim, output_dim)

        # 初始化 scale 接近 1，shift 接近 0
        nn.init.ones_(self.scale_projection.weight)
        nn.init.zeros_(self.scale_projection.bias)
        nn.init.zeros_(self.shift_projection.weight)
        nn.init.zeros_(self.shift_projection.bias)

    def forward(self, text_features, mode='bias'):
        """
        Args:
            text_features: (batch, text_dim) - 池化后的文本特征
            mode: 'bias' (加法) 或 'film' (FiLM 调制)

        Returns:
            如果 mode='bias': (batch, output_dim) - 直接加到 embedding
            如果 mode='film': (scale, shift) - FiLM 参数
        """
        if mode == 'bias':
            return self.global_projection(text_features)
        elif mode == 'film':
            scale = self.scale_projection(text_features)
            shift = self.shift_projection(text_features)
            return scale, shift
        else:
            raise ValueError(f"Unknown mode: {mode}")


class ConditionedEmbedding(nn.Module):
    """
    带条件注入的 Embedding 包装器

    替换 MuseFormer 的 embed_regular_tokens，在 embedding 后注入条件
    """
    def __init__(self, original_embedding, wrapper):
        super().__init__()
        self.original_embedding = original_embedding
        self.wrapper = wrapper  # 引用 ConditionedMuseFormerWrapper 来获取条件

        # 保持与原始 embedding 相同的接口
        self.padding_idx = original_embedding.padding_idx
        self.num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim

    def forward(self, tokens):
        """
        tokens: (seq_len, batch) 或 (batch, seq_len)
        返回: (seq_len, batch, dim) - 与原始一致
        """
        # 原始 embedding
        x = self.original_embedding(tokens)  # (seq_len, batch, dim)

        # 注入条件
        if self.wrapper.condition_bias is not None:
            # condition_bias: (batch, dim)
            # x: (seq_len, batch, dim)
            # 将 bias 加到每个位置的 embedding 上
            x = x + self.wrapper.condition_bias.unsqueeze(0)

        elif self.wrapper.condition_scale is not None:
            # FiLM: x = scale * x + shift
            x = self.wrapper.condition_scale.unsqueeze(0) * x + self.wrapper.condition_shift.unsqueeze(0)

        return x


class ConditionedMuseFormerWrapper(nn.Module):
    """
    条件化的 MuseFormer 包装器

    在 MuseFormer 的 embedding 层注入文本条件

    实现方式：
        用 ConditionedEmbedding 替换原始的 embed_regular_tokens
        这样在 MuseFormer 调用 embedding 时会自动注入条件
    """
    def __init__(self, museformer, condition_mode='bias'):
        super().__init__()

        self.museformer = museformer
        self.condition_mode = condition_mode

        # 保存原始 embedding 层的引用
        self.original_embed = museformer.decoder.embed_regular_tokens

        # 条件向量（由外部设置）
        self.condition_bias = None
        self.condition_scale = None
        self.condition_shift = None

        # 用带条件的 embedding 替换原始的
        self.conditioned_embed = ConditionedEmbedding(self.original_embed, self)
        museformer.decoder.embed_regular_tokens = self.conditioned_embed

    def set_condition(self, bias=None, scale=None, shift=None):
        """设置条件向量"""
        self.condition_bias = bias
        self.condition_scale = scale
        self.condition_shift = shift

    def clear_condition(self):
        """清除条件向量"""
        self.condition_bias = None
        self.condition_scale = None
        self.condition_shift = None

    def forward(self, src_tokens, **kwargs):
        """
        前向传播

        条件已经通过 ConditionedEmbedding 自动注入
        """
        return self.museformer(src_tokens, **kwargs)


class Text2MIDIFinal(nn.Module):
    """
    Text2MIDI 最终版本

    完整流程:
        1. 文本 → Qwen → 池化 → TextConditionEncoder → 条件向量
        2. MIDI tokens → (条件化的) MuseFormer → logits

    训练目标:
        最小化 cross-entropy loss: -log P(next_token | prev_tokens, text_condition)
    """
    def __init__(
        self,
        qwen_model_path,
        museformer_checkpoint_path,
        museformer_data_dir,
        condition_mode='bias',  # 'bias' 或 'film'
        hidden_dim=1024,
        freeze_encoder=True,
        freeze_decoder=True,
        device='cpu'
    ):
        super().__init__()

        self.condition_mode = condition_mode

        # 1. 文本编码器 (Qwen)
        print("加载 Qwen Encoder...")
        self.text_encoder = AutoModel.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        self.text_dim = self.text_encoder.config.hidden_size  # 1024

        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("  Qwen Encoder 已冻结")

        # 2. MuseFormer
        print("加载 MuseFormer...")
        museformer, args, dictionary = self._load_museformer(
            museformer_checkpoint_path, museformer_data_dir
        )
        self.museformer_dim = args.attention_embed_dim  # 512
        self.dictionary = dictionary

        # 包装 MuseFormer
        self.conditioned_museformer = ConditionedMuseFormerWrapper(
            museformer, condition_mode=condition_mode
        )

        if freeze_decoder:
            for param in self.conditioned_museformer.museformer.parameters():
                param.requires_grad = False
            print("  MuseFormer Decoder 已冻结")

        # 3. 文本条件编码器（需要训练）
        print("创建 TextConditionEncoder...")
        self.condition_encoder = TextConditionEncoder(
            text_dim=self.text_dim,
            output_dim=self.museformer_dim,
            hidden_dim=hidden_dim
        )

        # 特殊 token
        self.bos_idx = dictionary.bos()
        self.eos_idx = dictionary.eos()
        self.pad_idx = dictionary.pad()

        print(f"\n模型配置:")
        print(f"  Text dim: {self.text_dim}")
        print(f"  MuseFormer dim: {self.museformer_dim}")
        print(f"  Condition mode: {condition_mode}")
        print(f"  词表大小: {len(dictionary)}")

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

        print(f"  MuseFormer 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        return model, model_args, dictionary

    def encode_text(self, input_ids, attention_mask):
        """编码文本并生成条件向量"""
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 池化
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        # 生成条件
        if self.condition_mode == 'bias':
            condition = self.condition_encoder(pooled, mode='bias')
            return condition, None, None
        else:
            scale, shift = self.condition_encoder(pooled, mode='film')
            return None, scale, shift

    def forward(self, input_ids, attention_mask, midi_tokens, midi_mask=None):
        """
        训练时的前向传播

        Args:
            input_ids: 文本 token ids (batch, text_len)
            attention_mask: 文本 attention mask
            midi_tokens: MIDI token ids (batch, midi_len)
            midi_mask: MIDI padding mask

        Returns:
            logits: (batch, midi_len, vocab_size)
        """
        # 编码文本条件
        bias, scale, shift = self.encode_text(input_ids, attention_mask)

        # 设置条件
        self.conditioned_museformer.set_condition(bias=bias, scale=scale, shift=shift)

        # MuseFormer 前向传播
        logits, _ = self.conditioned_museformer(midi_tokens)

        # 清除条件
        self.conditioned_museformer.clear_condition()

        return logits

    @torch.no_grad()
    def generate(
        self,
        text,
        max_length=1024,
        temperature=1.0,
        top_k=8,
        top_p=0.9,
        device='cpu'
    ):
        """生成 MIDI"""
        self.eval()

        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        bias, scale, shift = self.encode_text(input_ids, attention_mask)
        self.conditioned_museformer.set_condition(bias=bias, scale=scale, shift=shift)

        # 生成
        generated = torch.LongTensor([[self.bos_idx]]).to(device)

        for step in range(max_length):
            logits, _ = self.conditioned_museformer(generated)
            next_logits = logits[0, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == self.eos_idx:
                break

            if (step + 1) % 100 == 0:
                print(f"  已生成 {step + 1} 个 tokens...")

        self.conditioned_museformer.clear_condition()

        token_list = generated[0].tolist()
        token_strings = [
            self.dictionary[t] for t in token_list
            if t not in [self.bos_idx, self.eos_idx, self.pad_idx]
        ]

        return token_list, token_strings

    def count_parameters(self):
        """统计参数"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        qwen_params = sum(p.numel() for p in self.text_encoder.parameters())
        museformer_params = sum(p.numel() for p in self.conditioned_museformer.museformer.parameters())
        condition_params = sum(p.numel() for p in self.condition_encoder.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'qwen': qwen_params,
            'museformer': museformer_params,
            'condition_encoder': condition_params,
        }


# ============================================
# 训练时间估算
# ============================================

def estimate_training_time(num_samples, batch_size=4, num_epochs=10, device_type='cpu'):
    """
    估算训练时间

    假设：
        - 只训练 TextConditionEncoder (~2M 参数)
        - Qwen 和 MuseFormer 都冻结

    由于只训练很小的部分，速度会很快
    """
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * num_epochs

    # 每步时间估算 (秒)
    time_per_step = {
        'cpu': 0.5,      # CPU 上主要是 inference
        'mps': 0.2,      # Apple Silicon
        'cuda': 0.05,    # NVIDIA GPU
    }

    step_time = time_per_step.get(device_type, 1.0)
    total_seconds = total_steps * step_time

    return {
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'hours': total_seconds / 3600,
        'days': total_seconds / 86400,
    }


# ============================================
# 模型配置
# ============================================

class ModelConfig:
    """模型配置"""
    qwen_model_path = "/Users/freeninglu/Desktop/MuseFormer/Qwen3-Embedding-0.6B"
    museformer_checkpoint = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
    museformer_data_dir = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/data-bin/lmd6remi"

    condition_mode = 'bias'  # 'bias' 或 'film'
    hidden_dim = 1024

    freeze_encoder = True
    freeze_decoder = True


def create_model(config=None, device='cpu'):
    """创建模型"""
    if config is None:
        config = ModelConfig()

    model = Text2MIDIFinal(
        qwen_model_path=config.qwen_model_path,
        museformer_checkpoint_path=config.museformer_checkpoint,
        museformer_data_dir=config.museformer_data_dir,
        condition_mode=config.condition_mode,
        hidden_dim=config.hidden_dim,
        freeze_encoder=config.freeze_encoder,
        freeze_decoder=config.freeze_decoder,
        device=device
    )

    return model


if __name__ == '__main__':
    print("=" * 60)
    print("Text2MIDI Final 模型测试")
    print("=" * 60)

    device = torch.device('cpu')
    config = ModelConfig()

    try:
        model = create_model(config, device=str(device))

        params = model.count_parameters()
        print(f"\n参数统计:")
        print(f"  总参数: {params['total'] / 1e6:.2f}M")
        print(f"  可训练: {params['trainable'] / 1e6:.2f}M (仅 Condition Encoder)")
        print(f"  冻结: {params['frozen'] / 1e6:.2f}M")
        print(f"  - Qwen: {params['qwen'] / 1e6:.2f}M")
        print(f"  - MuseFormer: {params['museformer'] / 1e6:.2f}M")
        print(f"  - Condition Encoder: {params['condition_encoder'] / 1e6:.2f}M")

        # 训练时间估算
        print("\n训练时间估算 (100,000 样本, batch=4, epochs=10):")
        for dev in ['cpu', 'mps', 'cuda']:
            est = estimate_training_time(100000, device_type=dev)
            print(f"  {dev}: {est['hours']:.1f} 小时 ({est['days']:.2f} 天)")

        print("\n✅ 模型创建成功!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
