"""
Text2MIDI 模型架构 v2
使用 Qwen3-Embedding 编码文本，MuseFormer 生成 MIDI

核心思路：
    MuseFormer 是一个语言模型，它通过预测下一个 token 来生成音乐。
    我们需要让它能够"看到"文本信息来条件化生成。

方案对比：
    1. Prefix Tuning: 将文本特征作为前缀 token 添加到输入序列
    2. Cross-Attention: 修改 MuseFormer 加入交叉注意力（侵入性强）
    3. Embedding Injection: 将文本特征注入到 token embedding 中

本文件实现 Prefix Tuning 方案，因为：
    - 不需要修改 MuseFormer 内部结构
    - 训练简单，只需要训练投影层
    - 文本特征作为"软提示"引导生成
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import math

# 添加 MuseFormer 路径
MUSEFORMER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'muzic', 'museformer')
sys.path.insert(0, MUSEFORMER_PATH)


class PrefixEncoder(nn.Module):
    """
    Prefix 编码器
    将文本特征转换为 MuseFormer 的前缀 embedding

    思路：
        1. Qwen 编码文本 → (batch, text_len, 1024)
        2. 池化 → (batch, 1024)
        3. 投影并扩展为多个前缀 token → (batch, prefix_len, 512)
    """
    def __init__(
        self,
        input_dim=1024,
        output_dim=512,
        prefix_length=32,
        hidden_dim=1024,
        num_layers=2
    ):
        super().__init__()

        self.prefix_length = prefix_length
        self.output_dim = output_dim

        # 投影层：将池化后的文本特征映射到 prefix
        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim

        # 最后一层输出 prefix_length * output_dim
        layers.append(nn.Linear(current_dim, prefix_length * output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, text_hidden):
        """
        Args:
            text_hidden: (batch, hidden_dim) - 池化后的文本特征
        Returns:
            prefix: (batch, prefix_length, output_dim)
        """
        batch_size = text_hidden.size(0)
        prefix = self.mlp(text_hidden)  # (batch, prefix_len * output_dim)
        prefix = prefix.view(batch_size, self.prefix_length, self.output_dim)
        return prefix


class Text2MIDIWithPrefix(nn.Module):
    """
    Text2MIDI 使用 Prefix Tuning

    架构：
        文本 → Qwen → 池化 → PrefixEncoder → Prefix Tokens
                                                  ↓
        [Prefix] + [BOS] + [MIDI tokens] → MuseFormer → Next Token Prediction

    训练策略：
        1. 冻结 Qwen（文本编码器）
        2. 冻结 MuseFormer（MIDI 生成器）
        3. 只训练 PrefixEncoder（学习文本到音乐的映射）

    这样做的好处：
        - 参数效率高：只训练 ~2M 参数
        - 不破坏 MuseFormer 的音乐生成能力
        - 前缀作为"软提示"引导风格/情绪
    """
    def __init__(
        self,
        qwen_model_path,
        museformer_checkpoint_path,
        museformer_data_dir,
        prefix_length=32,
        projection_hidden_dim=1024,
        freeze_encoder=True,
        freeze_decoder=True,
        device='cpu'
    ):
        super().__init__()

        self.device = device
        self.prefix_length = prefix_length

        # 1. 加载 Qwen 文本编码器
        print("加载 Qwen Encoder...")
        self.text_encoder = AutoModel.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        qwen_dim = self.text_encoder.config.hidden_size  # 1024

        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("  Qwen Encoder 已冻结")

        # 2. 加载 MuseFormer
        print("加载 MuseFormer...")
        self.museformer, self.museformer_args, self.dictionary = self._load_museformer(
            museformer_checkpoint_path, museformer_data_dir
        )
        museformer_dim = self.museformer_args.attention_embed_dim  # 512

        if freeze_decoder:
            for param in self.museformer.parameters():
                param.requires_grad = False
            print("  MuseFormer Decoder 已冻结")

        # 3. Prefix 编码器（唯一需要训练的部分）
        print("创建 Prefix Encoder...")
        self.prefix_encoder = PrefixEncoder(
            input_dim=qwen_dim,
            output_dim=museformer_dim,
            prefix_length=prefix_length,
            hidden_dim=projection_hidden_dim
        )

        # 特殊 token
        self.bos_idx = self.dictionary.bos()
        self.eos_idx = self.dictionary.eos()
        self.pad_idx = self.dictionary.pad()

        # MuseFormer 的 embedding 层
        self.midi_embedding = self.museformer.decoder.embed_regular_tokens
        self.embed_scale = self.museformer.decoder.embed_scale

        print(f"\n模型配置:")
        print(f"  Qwen dim: {qwen_dim}")
        print(f"  MuseFormer dim: {museformer_dim}")
        print(f"  Prefix length: {prefix_length}")
        print(f"  词表大小: {len(self.dictionary)}")

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
        """
        编码文本并生成前缀

        Returns:
            prefix: (batch, prefix_length, museformer_dim)
        """
        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 使用 [CLS] 或平均池化
        hidden = outputs.last_hidden_state  # (batch, seq_len, 1024)

        # 平均池化
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, 1024)

        # 生成前缀
        prefix = self.prefix_encoder(pooled)  # (batch, prefix_len, 512)

        return prefix

    def forward(self, input_ids, attention_mask, midi_tokens, midi_mask=None):
        """
        训练时的前向传播

        问题：MuseFormer 的 forward 接口比较复杂，需要 chunk_points 等信息
        简化方案：直接使用 MuseFormer 的 forward，暂不处理 prefix

        后续改进：修改 MuseFormer 的 embedding 层，将 prefix 注入
        """
        # 生成前缀（暂时不使用，需要进一步集成）
        prefix = self.encode_text(input_ids, attention_mask)

        # 使用 MuseFormer 前向传播
        logits, _ = self.museformer(midi_tokens)

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
        """
        条件化生成 MIDI

        当前简化实现：直接使用 MuseFormer 生成
        TODO: 将 prefix 注入到生成过程
        """
        self.eval()

        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 生成前缀（当前版本暂未使用）
        prefix = self.encode_text(input_ids, attention_mask)

        # 从 BOS 开始生成
        generated = torch.LongTensor([[self.bos_idx]]).to(device)

        for step in range(max_length):
            logits, _ = self.museformer(generated)
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
        frozen = total - trainable

        prefix_params = sum(p.numel() for p in self.prefix_encoder.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'prefix_encoder': prefix_params,
        }


# ============================================
# 使用对比学习的方案
# ============================================

class ContrastiveText2MIDI(nn.Module):
    """
    对比学习方案

    思路：
        1. 用 Qwen 编码文本 → text_embedding
        2. 用 MuseFormer 的 encoder 编码 MIDI → midi_embedding
        3. 训练投影层使得配对的 (text, midi) embedding 接近

    训练完成后：
        1. 输入文本 → text_embedding
        2. 在 MIDI embedding 空间中找最近邻
        3. 或者用 text_embedding 引导 MuseFormer 生成

    这是一种检索+生成的混合方案
    """
    def __init__(
        self,
        qwen_model_path,
        museformer_checkpoint_path,
        museformer_data_dir,
        embedding_dim=512,
        device='cpu'
    ):
        super().__init__()

        # 加载 Qwen
        print("加载 Qwen...")
        self.text_encoder = AutoModel.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
        qwen_dim = self.text_encoder.config.hidden_size

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # 加载 MuseFormer
        print("加载 MuseFormer...")
        self.museformer, self.args, self.dictionary = self._load_museformer(
            museformer_checkpoint_path, museformer_data_dir
        )
        museformer_dim = self.args.attention_embed_dim

        for param in self.museformer.parameters():
            param.requires_grad = False

        # 文本投影层
        self.text_projection = nn.Sequential(
            nn.Linear(qwen_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # MIDI 投影层
        self.midi_projection = nn.Sequential(
            nn.Linear(museformer_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # 温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        self.bos_idx = self.dictionary.bos()
        self.eos_idx = self.dictionary.eos()
        self.pad_idx = self.dictionary.pad()

    def _load_museformer(self, checkpoint_path, data_dir):
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

    def encode_text(self, input_ids, attention_mask):
        """编码文本"""
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        text_emb = self.text_projection(pooled)
        text_emb = F.normalize(text_emb, dim=-1)

        return text_emb

    def encode_midi(self, midi_tokens):
        """编码 MIDI"""
        with torch.no_grad():
            # 使用 MuseFormer 的 features_only 模式
            features, _ = self.museformer(midi_tokens, features_only=True)

        # 使用最后一个非 padding 位置的特征
        # 简化：使用平均池化
        midi_emb = features.mean(dim=1)  # (batch, dim)

        midi_emb = self.midi_projection(midi_emb)
        midi_emb = F.normalize(midi_emb, dim=-1)

        return midi_emb

    def forward(self, input_ids, attention_mask, midi_tokens):
        """
        对比学习的前向传播

        Returns:
            loss: InfoNCE loss
        """
        text_emb = self.encode_text(input_ids, attention_mask)  # (batch, dim)
        midi_emb = self.encode_midi(midi_tokens)  # (batch, dim)

        # 计算相似度矩阵
        logits = torch.matmul(text_emb, midi_emb.T) / self.temperature

        # InfoNCE loss
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        loss_t2m = F.cross_entropy(logits, labels)  # text -> midi
        loss_m2t = F.cross_entropy(logits.T, labels)  # midi -> text

        loss = (loss_t2m + loss_m2t) / 2

        return loss, text_emb, midi_emb


# ============================================
# 配置
# ============================================

class ModelConfigV2:
    """模型配置 v2"""
    qwen_model_path = "/Users/freeninglu/Desktop/MuseFormer/Qwen3-Embedding-0.6B"
    museformer_checkpoint = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
    museformer_data_dir = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/data-bin/lmd6remi"

    prefix_length = 32
    projection_hidden_dim = 1024
    embedding_dim = 512

    freeze_encoder = True
    freeze_decoder = True


if __name__ == '__main__':
    print("=" * 60)
    print("Text2MIDI v2 模型测试")
    print("=" * 60)

    device = torch.device('cpu')
    config = ModelConfigV2()

    print("\n测试 Prefix 模型...")
    try:
        model = Text2MIDIWithPrefix(
            qwen_model_path=config.qwen_model_path,
            museformer_checkpoint_path=config.museformer_checkpoint,
            museformer_data_dir=config.museformer_data_dir,
            prefix_length=config.prefix_length,
            projection_hidden_dim=config.projection_hidden_dim,
            freeze_encoder=config.freeze_encoder,
            freeze_decoder=config.freeze_decoder,
            device=str(device)
        )

        params = model.count_parameters()
        print(f"\n参数统计:")
        print(f"  总参数: {params['total'] / 1e6:.2f}M")
        print(f"  可训练: {params['trainable'] / 1e6:.2f}M (仅 Prefix Encoder)")
        print(f"  冻结: {params['frozen'] / 1e6:.2f}M")
        print(f"  Prefix Encoder: {params['prefix_encoder'] / 1e6:.2f}M")

        print("\n✅ Prefix 模型创建成功!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
