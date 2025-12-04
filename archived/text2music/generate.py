#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2Music 推理和生成脚本

使用训练好的模型从文本生成 MIDI：
1. 文本 → Summary Token (Diffusion Bridge)
2. Summary Token → MIDI (HID-MeloFormer Hook 注入)

用法:
    python generate.py \
        --text2summary-model outputs/text2summary/best.pt \
        --meloformer-model checkpoints/hid_meloformer.pt \
        --text "A melodic piano piece in C major" \
        --output output.mid
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

import torch
from transformers import AutoTokenizer

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.text2summary import Text2SummaryModel
from hid_museformer_v0.8.model.hid_museformer import HIDMuseFormer, create_model as create_meloformer
from hid_museformer_v0.8.data.txt_to_midi import tokens_to_midi


class Text2MusicGenerator:
    """
    Text2Music 生成器

    整合 Text2Summary 和 HID-MeloFormer 实现端到端生成
    """

    def __init__(
        self,
        text2summary_model: Text2SummaryModel,
        meloformer_model: HIDMuseFormer,
        tokenizer,
        vocab: dict,
        device: torch.device = None,
        injection_layer: int = 3,
    ):
        """
        Args:
            text2summary_model: 训练好的 Text2Summary 模型
            meloformer_model: 训练好的 HID-MeloFormer 模型
            tokenizer: 文本 tokenizer
            vocab: HID-MeloFormer 词汇表
            device: 设备
            injection_layer: Summary Token 注入层
        """
        self.text2summary = text2summary_model
        self.meloformer = meloformer_model
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.injection_layer = injection_layer

        self.text2summary.to(self.device)
        self.meloformer.to(self.device)
        self.text2summary.eval()
        self.meloformer.eval()

        # 构建反向词汇表
        self.idx_to_token = {v: k for k, v in vocab.items()}

    @torch.no_grad()
    def generate_summary_tokens(
        self,
        texts: List[str],
        num_bars: int = 16,
        steps: int = 20,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        从文本生成 Summary Token

        Args:
            texts: 文本列表
            num_bars: 生成的 bar 数量
            steps: ODE 采样步数
            cfg_scale: CFG 强度

        Returns:
            summary_tokens: (batch, num_bars, 512)
        """
        return self.text2summary.generate_from_text(
            texts=texts,
            tokenizer=self.tokenizer,
            num_bars=num_bars,
            steps=steps,
            cfg_scale=cfg_scale,
            device=self.device,
        )

    @torch.no_grad()
    def generate_midi_tokens(
        self,
        summary_tokens: torch.Tensor,
        prompt_tokens: Optional[torch.Tensor] = None,
        max_length: int = 8192,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[int]:
        """
        使用 Summary Token 生成 MIDI Token

        通过 Hook 注入机制将生成的 Summary Token 注入 MeloFormer

        Args:
            summary_tokens: (1, num_bars, 512) 生成的 Summary Token
            prompt_tokens: 可选的 prompt tokens
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus sampling p
            top_k: top-k sampling k

        Returns:
            token_ids: 生成的 token ID 列表
        """
        # 准备注入的 Summary Token
        injected_summary = [summary_tokens]
        injection_counter = [0]

        def inject_hook(module, input, output):
            """Hook: 替换 Summary Token"""
            sum_out, reg_out = output

            # 返回注入的 Summary Token
            # 注意: 可能需要调整形状以匹配当前生成位置
            return (injected_summary[0], reg_out)

        # 注册 hook
        handle = self.meloformer.layers[self.injection_layer].register_forward_hook(inject_hook)

        try:
            # 初始化
            if prompt_tokens is None:
                # 使用 BOS token
                bos_id = self.vocab.get('<BOS>', self.vocab.get('<s>', 0))
                generated = [bos_id]
            else:
                generated = prompt_tokens.tolist() if isinstance(prompt_tokens, torch.Tensor) else list(prompt_tokens)

            # 自回归生成
            for _ in range(max_length):
                # 准备输入
                input_ids = torch.tensor([generated], device=self.device)

                # 简化：不使用完整的 mask 信息 (用于演示)
                # 实际使用需要正确构建 chord_ids, instrument_ids 等
                logits = self.meloformer(input_ids)  # (1, seq_len, vocab_size)

                # 取最后一个位置的 logits
                next_logits = logits[0, -1, :] / temperature

                # Top-k 过滤
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float('-inf')

                # 采样
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # 检查 EOS
                eos_id = self.vocab.get('<EOS>', self.vocab.get('</s>', 1))
                if next_token == eos_id:
                    break

                generated.append(next_token)

            return generated

        finally:
            handle.remove()

    def tokens_to_text(self, token_ids: List[int]) -> str:
        """将 token IDs 转换为文本表示"""
        return ' '.join(self.idx_to_token.get(tid, f'[{tid}]') for tid in token_ids)

    def generate(
        self,
        text: str,
        num_bars: int = 16,
        output_path: Optional[str] = None,
        summary_steps: int = 20,
        cfg_scale: float = 1.0,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> dict:
        """
        端到端生成：文本 → MIDI

        Args:
            text: 输入文本描述
            num_bars: 生成的 bar 数量
            output_path: MIDI 输出路径
            summary_steps: Summary Token 生成步数
            cfg_scale: CFG 强度
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            verbose: 是否打印进度

        Returns:
            {
                'text': str,
                'summary_tokens': Tensor,
                'midi_tokens': List[int],
                'midi_path': str (如果保存了文件),
            }
        """
        if verbose:
            print(f"Input: {text}")
            print(f"Generating {num_bars} bars...")

        # Step 1: 生成 Summary Token
        if verbose:
            print("  Step 1: Generating Summary Tokens...")

        summary_tokens = self.generate_summary_tokens(
            texts=[text],
            num_bars=num_bars,
            steps=summary_steps,
            cfg_scale=cfg_scale,
        )

        if verbose:
            print(f"    Summary shape: {summary_tokens.shape}")

        # Step 2: 生成 MIDI Tokens
        if verbose:
            print("  Step 2: Generating MIDI Tokens...")

        midi_tokens = self.generate_midi_tokens(
            summary_tokens=summary_tokens,
            max_length=max_tokens,
            temperature=temperature,
        )

        if verbose:
            print(f"    Generated {len(midi_tokens)} tokens")

        result = {
            'text': text,
            'summary_tokens': summary_tokens,
            'midi_tokens': midi_tokens,
        }

        # Step 3: 保存 MIDI
        if output_path is not None:
            if verbose:
                print(f"  Step 3: Saving MIDI to {output_path}...")

            try:
                # 转换为 MIDI
                token_text = self.tokens_to_text(midi_tokens)
                midi = tokens_to_midi(token_text, output_path)
                result['midi_path'] = output_path

                if verbose:
                    print(f"    Saved: {output_path}")
            except Exception as e:
                if verbose:
                    print(f"    Error saving MIDI: {e}")

        return result


def load_generator(
    text2summary_path: str,
    meloformer_path: str,
    text_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: Optional[torch.device] = None,
) -> Text2MusicGenerator:
    """
    加载生成器

    Args:
        text2summary_path: Text2Summary 模型路径
        meloformer_path: HID-MeloFormer 模型路径
        text_model_name: 文本编码器模型名
        device: 设备

    Returns:
        Text2MusicGenerator 实例
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 Text2Summary
    t2s_checkpoint = torch.load(text2summary_path, map_location='cpu')
    t2s_config = t2s_checkpoint.get('config', {}).get('model', {})

    text2summary = Text2SummaryModel(
        text_encoder_name=t2s_config.get('text_encoder', text_model_name),
        summary_dim=t2s_config.get('summary_dim', 512),
        bridge_hidden_dim=t2s_config.get('bridge_hidden_dim', 1024),
        bridge_num_layers=t2s_config.get('bridge_num_layers', 6),
        freeze_text_encoder=True,
        use_transformer_bridge=t2s_config.get('use_transformer_bridge', False),
    )
    text2summary.load_state_dict(t2s_checkpoint['model_state_dict'])

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        t2s_config.get('text_encoder', text_model_name),
        trust_remote_code=True,
    )

    # 加载 MeloFormer
    mf_checkpoint = torch.load(meloformer_path, map_location='cpu')
    meloformer = create_meloformer(
        vocab_size=mf_checkpoint.get('vocab_size', 643),
        model_size=mf_checkpoint.get('model_size', 'base'),
    )
    meloformer.load_state_dict(mf_checkpoint['model_state_dict'])

    # 加载词汇表
    vocab = mf_checkpoint.get('vocab', {})
    if not vocab:
        # 默认词汇表
        vocab = {f'token_{i}': i for i in range(643)}

    return Text2MusicGenerator(
        text2summary_model=text2summary,
        meloformer_model=meloformer,
        tokenizer=tokenizer,
        vocab=vocab,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Generate MIDI from text')
    parser.add_argument('--text2summary-model', required=True, help='Text2Summary model path')
    parser.add_argument('--meloformer-model', required=True, help='HID-MeloFormer model path')
    parser.add_argument('--text', required=True, help='Input text description')
    parser.add_argument('--output', default='output.mid', help='Output MIDI path')
    parser.add_argument('--num-bars', type=int, default=16, help='Number of bars to generate')
    parser.add_argument('--steps', type=int, default=20, help='ODE sampling steps')
    parser.add_argument('--cfg-scale', type=float, default=1.0, help='CFG scale')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Max tokens to generate')
    args = parser.parse_args()

    # 加载生成器
    print("Loading models...")
    generator = load_generator(args.text2summary_model, args.meloformer_model)

    # 生成
    result = generator.generate(
        text=args.text,
        num_bars=args.num_bars,
        output_path=args.output,
        summary_steps=args.steps,
        cfg_scale=args.cfg_scale,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        verbose=True,
    )

    print("\nDone!")
    print(f"Generated {len(result['midi_tokens'])} tokens")
    if 'midi_path' in result:
        print(f"MIDI saved to: {result['midi_path']}")


if __name__ == '__main__':
    main()
