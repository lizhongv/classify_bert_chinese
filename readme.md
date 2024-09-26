https://zhuanlan.zhihu.com/p/524487313


## 输入

文本 → CLS + tokens + SEP → ids → word embedding + pos embedding → hidden_states 

[batch_size, seq_length, hidden_size]

- [CLS]：表示整句话的语义表示
- [UNK]：输入句子的token不在词表中时，使用UNK embedding来表示
- [SEP]：用来区分两个句子或表示句子结尾
- [PAD]：预训练模型接收长度相同的batch data，因此短句可通过padding来对齐
- [MASK]：主要为mlm任务

input_ids 

token_type_ids：token类型，若句子对，属于第一个句子token则此id设为0，属于第二个句子token则此id设为1

attention_mask：实际文本内容的token设为1，填充部分token设为0

## BERT tokenizer
分词方法，采用WordPiece模型，将每个词分解为subword，来解决一词多义、拼写变化等问题。

主要两个分词器：先进行`BasicTokenizer`得到一个分的比较粗的token列表，然后再对每个token进行一次` WordpieceTokenizer`，得到最终的分词结果。
- BasicTokenizer：初步分词，主要用于去除奇怪字符、空格分词、多余字符、标点分词

- WordpieceTokenizer：进一步切分，得到subword，以`##`开头

## BERT pos embedding
BERT中位置编码是可训练的

在于BERT使用了随机初始化训练出来的绝对位置编码，最大位置设为为512，若是文本长于512便无位置编码可用。

## OOV问题
由于模型在特定语料库上进行训练的，词表也是固定的。
当应用到其他数据上时，可能出现某些token不在词表中，通常称为词汇不足问题（OOV）

## mask

```python
attention_mask # [32, 1, 512]
if attention_mask is None:
    attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
if attention_mask.dim() == 3:
    extended_attention_mask = attention_mask[:, None, :, :] # [32, 1, 1, 512]

extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

def get_extended_attention_mask(
    self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None) -> Tensor:
    
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask # 将pad处赋值为浮点数的最小值


```

attention_mask + 双向

## pooler 

[batch, seq_len, dim]