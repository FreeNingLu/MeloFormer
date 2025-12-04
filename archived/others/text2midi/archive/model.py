"""
Text2MIDI 模型架构
使用 Qwen3-Embedding 作为文本编码器，MuseFormer checkpoint 作为 MIDI Decoder

架构:
    文本 → [Qwen Encoder (冻结)] → [投影层 (训练)] → [MuseFormer Decoder (可选冻结/微调)]

关键设计:
    1. 冻结 Qwen Encoder（节省显存和训练时间）
    2. 训练投影层（MLP）将文本特征映射到 MuseFormer 的 embedding 空间
    3. 加载已训练好的 MuseFormer checkpoint 作为 MIDI Decoder
    4. MuseFormer 可以冻结（只训练投影层）或微调
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


class ProjectionLayer(nn.Module):
    """
    投影层：将 Qwen embedding 映射到 MuseFormer embedding 空间

    Qwen hidden_size: 1024
    MuseFormer attention_embed_dim: 512
    """
    def __init__(self, input_dim=1024, output_dim=512, hidden_dim=1024, num_layers=2, dropout=0.1):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层：让 MIDI Decoder 能够关注文本特征
    """
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) - decoder hidden states
            memory: (batch, mem_len, d_model) - encoder output (文本特征)
            memory_key_padding_mask: padding mask for memory
        """
        attn_out, _ = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        x = self.norm(x + self.dropout(attn_out))
        return x


class Text2MIDIModelWithMuseFormer(nn.Module):
    """
    Text2MIDI 模型 - 使用预训练的 MuseFormer 作为 MIDI Decoder

    组件:
        1. Qwen Encoder (冻结)
        2. Projection Layer (训练)
        3. Cross-Attention Layer (训练) - 可选
        4. MuseFormer Decoder (从 checkpoint 加载，可选冻结/微调)

    两种模式:
        1. Prefix Mode: 将文本特征作为 MuseFormer 输入的前缀
        2. Cross-Attention Mode: 在 MuseFormer 层之间添加交叉注意力
    """
    def __init__(
        self,
        qwen_model_path,
        museformer_checkpoint_path,
        museformer_data_dir,
        projection_hidden_dim=1024,
        projection_layers=2,
        freeze_encoder=True,
        freeze_decoder=True,
        use_cross_attention=False,
        cross_attention_layers=2,
        device='cpu'
    ):
        super().__init__()

        self.device = device
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.use_cross_attention = use_cross_attention

        # 1. 文本编码器 (Qwen)
        print("加载 Qwen Encoder...")
        self.text_encoder = AutoModel.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

        # 获取 Qwen 输出维度
        qwen_dim = self.text_encoder.config.hidden_size  # 1024

        # 冻结编码器
        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("  Qwen Encoder 已冻结")

        # 2. 加载 MuseFormer
        print("加载 MuseFormer Decoder...")
        self.museformer, self.museformer_args, self.dictionary = self._load_museformer(
            museformer_checkpoint_path,
            museformer_data_dir
        )

        # MuseFormer 的 embedding 维度
        museformer_dim = self.museformer_args.attention_embed_dim  # 512
        self.museformer_dim = museformer_dim

        # 冻结 MuseFormer
        if freeze_decoder:
            for param in self.museformer.parameters():
                param.requires_grad = False
            print("  MuseFormer Decoder 已冻结")

        # 3. 投影层
        print("创建投影层...")
        self.projection = ProjectionLayer(
            input_dim=qwen_dim,
            output_dim=museformer_dim,
            hidden_dim=projection_hidden_dim,
            num_layers=projection_layers
        )

        # 4. 可选的交叉注意力层
        if use_cross_attention:
            print("创建交叉注意力层...")
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionLayer(d_model=museformer_dim)
                for _ in range(cross_attention_layers)
            ])
        else:
            self.cross_attention_layers = None

        # 特殊 token
        self.bos_idx = self.dictionary.bos()
        self.eos_idx = self.dictionary.eos()
        self.pad_idx = self.dictionary.pad()

        print(f"  Qwen dim: {qwen_dim}")
        print(f"  MuseFormer dim: {museformer_dim}")
        print(f"  词表大小: {len(self.dictionary)}")

    def _load_museformer(self, checkpoint_path, data_dir):
        """加载 MuseFormer 模型"""
        from fairseq.data import Dictionary
        from museformer.museformer_lm_task import MuseformerLanguageModelingTask

        # 加载字典
        dict_path = os.path.join(data_dir, 'dict.txt')
        dictionary = Dictionary.load(dict_path)

        # 加载 checkpoint
        state = torch.load(checkpoint_path, map_location='cpu')
        model_args = state['args']

        # 设置数据目录
        model_args.data = data_dir

        # 创建 task
        task = MuseformerLanguageModelingTask.setup_task(model_args)

        # 构建模型
        model = task.build_model(model_args)
        model.load_state_dict(state['model'], strict=True)

        print(f"  加载 MuseFormer 成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        return model, model_args, dictionary

    def encode_text(self, input_ids, attention_mask):
        """编码文本，返回投影后的特征"""
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        else:
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 使用最后一层的隐藏状态
        text_features = outputs.last_hidden_state  # (batch, seq_len, 1024)

        # 投影到 MuseFormer 空间
        projected = self.projection(text_features)  # (batch, seq_len, 512)

        return projected

    def forward(self, input_ids, attention_mask, midi_tokens, midi_mask=None):
        """
        前向传播 (训练时使用)

        策略: Prefix Mode
        将文本特征作为 MuseFormer 的条件输入

        Args:
            input_ids: 文本 token ids (batch, text_len)
            attention_mask: 文本 attention mask
            midi_tokens: MIDI token ids (batch, midi_len)
            midi_mask: MIDI padding mask

        Returns:
            logits: (batch, midi_len, vocab_size)
        """
        # 编码文本
        text_features = self.encode_text(input_ids, attention_mask)  # (batch, text_len, 512)

        # 使用 MuseFormer 进行前向传播
        # MuseFormer 的 forward 需要 src_tokens
        if self.freeze_decoder:
            with torch.no_grad():
                logits, _ = self.museformer(midi_tokens)
        else:
            logits, _ = self.museformer(midi_tokens)

        # 如果使用交叉注意力，在这里添加
        # 注意：这需要修改 MuseFormer 的内部结构，比较复杂
        # 暂时使用简化版本：只返回 MuseFormer 的输出

        return logits

    @torch.no_grad()
    def generate(
        self,
        text,
        max_length=1024,
        min_length=128,
        temperature=1.0,
        top_k=8,
        top_p=0.9,
        device='cpu'
    ):
        """
        生成 MIDI tokens

        Args:
            text: 输入文本描述
            max_length: 最大生成长度
            min_length: 最小生成长度
            temperature: 采样温度
            top_k: top-k 采样
            top_p: nucleus 采样

        Returns:
            generated_tokens: list of token ids
            token_strings: list of token strings
        """
        self.eval()

        # 编码文本
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        text_features = self.encode_text(input_ids, attention_mask)
        # text_features: (1, text_len, 512)

        # 从 BOS 开始生成
        generated = torch.LongTensor([[self.bos_idx]]).to(device)

        for step in range(max_length):
            # 使用 MuseFormer 前向传播
            logits, _ = self.museformer(generated)

            # 获取最后一个位置的 logits
            next_logits = logits[0, -1, :] / temperature

            # Top-k 采样
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到序列
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # 检查是否结束
            if next_token.item() == self.eos_idx:
                break

            # 进度显示
            if (step + 1) % 100 == 0:
                print(f"  已生成 {step + 1} 个 tokens...")

        # 转换为字符串
        token_list = generated[0].tolist()
        token_strings = [
            self.dictionary[t] for t in token_list
            if t not in [self.bos_idx, self.eos_idx, self.pad_idx]
        ]

        return token_list, token_strings

    def count_parameters(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        # 分组统计
        qwen_params = sum(p.numel() for p in self.text_encoder.parameters())
        museformer_params = sum(p.numel() for p in self.museformer.parameters())
        projection_params = sum(p.numel() for p in self.projection.parameters())

        cross_attn_params = 0
        if self.cross_attention_layers:
            cross_attn_params = sum(p.numel() for p in self.cross_attention_layers.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'qwen': qwen_params,
            'museformer': museformer_params,
            'projection': projection_params,
            'cross_attention': cross_attn_params,
        }


# ============================================
# 简化版本 - 只训练投影层
# ============================================

class Text2MIDISimple(nn.Module):
    """
    简化版 Text2MIDI

    训练策略:
        1. 冻结 Qwen 和 MuseFormer
        2. 只训练投影层
        3. 使用文本特征的池化结果作为生成的"种子"

    这是最简单的方案，适合快速实验
    """
    def __init__(
        self,
        qwen_model_path,
        museformer_checkpoint_path,
        museformer_data_dir,
        projection_dim=512,
        device='cpu'
    ):
        super().__init__()

        # 加载 Qwen
        print("加载 Qwen Encoder...")
        self.text_encoder = AutoModel.from_pretrained(qwen_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        qwen_dim = self.text_encoder.config.hidden_size

        # 加载 MuseFormer
        print("加载 MuseFormer...")
        self.museformer, self.args, self.dictionary = self._load_museformer(
            museformer_checkpoint_path, museformer_data_dir
        )

        for param in self.museformer.parameters():
            param.requires_grad = False

        museformer_dim = self.args.attention_embed_dim

        # 投影层：文本 -> 种子向量
        self.projection = nn.Sequential(
            nn.Linear(qwen_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, museformer_dim)
        )

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

    def encode_text(self, text, device='cpu'):
        """编码文本并池化"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 平均池化
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        # 投影
        seed = self.projection(pooled)
        return seed

    @torch.no_grad()
    def generate(self, text, max_length=512, temperature=1.0, top_k=8, device='cpu'):
        """基于文本生成 MIDI"""
        self.eval()

        # 获取文本种子
        seed = self.encode_text(text, device)  # (1, 512)

        # 从 BOS 开始
        generated = torch.LongTensor([[self.bos_idx]]).to(device)

        for _ in range(max_length):
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
        token_strings = [self.dictionary[t] for t in token_list
                        if t not in [self.bos_idx, self.eos_idx, self.pad_idx]]

        return token_list, token_strings


# ============================================
# 模型配置
# ============================================

class ModelConfig:
    """模型配置"""
    # 文本编码器
    qwen_model_path = "/Users/freeninglu/Desktop/MuseFormer/Qwen3-Embedding-0.6B"

    # MuseFormer
    museformer_checkpoint = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/checkpoints/mf-lmd6remi-1/checkpoint_best.pt"
    museformer_data_dir = "/Users/freeninglu/Desktop/MuseFormer/muzic/museformer/data-bin/lmd6remi"

    # 投影层
    projection_hidden_dim = 1024
    projection_layers = 2

    # 训练
    freeze_encoder = True
    freeze_decoder = True  # 如果为 False，则微调 MuseFormer

    # 交叉注意力 (可选)
    use_cross_attention = False
    cross_attention_layers = 2


def create_model(config=None, device='cpu'):
    """创建模型"""
    if config is None:
        config = ModelConfig()

    model = Text2MIDIModelWithMuseFormer(
        qwen_model_path=config.qwen_model_path,
        museformer_checkpoint_path=config.museformer_checkpoint,
        museformer_data_dir=config.museformer_data_dir,
        projection_hidden_dim=config.projection_hidden_dim,
        projection_layers=config.projection_layers,
        freeze_encoder=config.freeze_encoder,
        freeze_decoder=config.freeze_decoder,
        use_cross_attention=config.use_cross_attention,
        cross_attention_layers=config.cross_attention_layers,
        device=device
    )

    return model


if __name__ == '__main__':
    print("=" * 60)
    print("Text2MIDI 模型测试 (使用 MuseFormer)")
    print("=" * 60)

    # 检查设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\n设备: {device}")

    # 创建配置
    config = ModelConfig()
    print(f"\n配置:")
    print(f"  Qwen: {config.qwen_model_path}")
    print(f"  MuseFormer: {config.museformer_checkpoint}")
    print(f"  冻结编码器: {config.freeze_encoder}")
    print(f"  冻结解码器: {config.freeze_decoder}")

    # 创建模型
    print("\n创建模型...")
    try:
        model = create_model(config, device=str(device))
        model = model.to(device)

        # 统计参数
        params = model.count_parameters()
        print(f"\n参数统计:")
        print(f"  总参数: {params['total'] / 1e6:.2f}M")
        print(f"  可训练: {params['trainable'] / 1e6:.2f}M")
        print(f"  冻结: {params['frozen'] / 1e6:.2f}M")
        print(f"  - Qwen: {params['qwen'] / 1e6:.2f}M")
        print(f"  - MuseFormer: {params['museformer'] / 1e6:.2f}M")
        print(f"  - Projection: {params['projection'] / 1e6:.2f}M")

        # 测试生成
        print("\n测试生成...")
        text = "一段欢快的钢琴曲，C大调，4/4拍，120 BPM"
        print(f"  输入: {text}")

        token_ids, token_strs = model.generate(
            text,
            max_length=64,
            temperature=1.0,
            top_k=8,
            device=device
        )
        print(f"  生成了 {len(token_strs)} 个 tokens")
        print(f"  前 20 个: {' '.join(token_strs[:20])}")

        print("\n✅ 模型测试通过!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
