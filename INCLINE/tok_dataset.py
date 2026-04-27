import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """
    将文本样本数据集转换为由分词器 (tokenizer) 转换的 token 序列数据集。
    这些 token 带有 position ids 和 attention masks，可以直接提供给模型。
    """

    def __init__(self, text_dataset, tokenizer=None, maxlen=None, field="text"):
        self.text_dataset = text_dataset
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        text = self.text_dataset[i]
        if self.field is not None:
            text = text[self.field]
        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )
        position_ids = list(range(len(token_list)))
        attention_mask = [1] * len(token_list)
        return dict(
            input_ids=torch.tensor(token_list),
            position_ids=torch.tensor(position_ids),
            attention_mask=torch.tensor(attention_mask),
        )


def dict_to_(data, device):
    """
    将张量 (tensor) 字典移动到指定的设备。
    """
    for k in data:
        data[k] = data[k].to(device)
    return data


def length_collation(token_size):
    """
    对一批序列进行排序，并将其分解为相同大小序列的子批次 (subbatches)，
    根据需要进行填充 (padding)。每个批次包含的 token 总数不超过 token_size 
    (如果单个序列较大，则包含该单个序列)。
    """

    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = 0
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches

    return collate_fn


def make_padded_batch(items):
    """
    在批次中填充序列，使它们都与最长的序列长度相同。
    """
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    return {
        k: pad_sequence([d[k] for d in items if len(d["input_ids"])], batch_first=True)
        for k, v in items[0].items()
    }


def flatten_masked_batch(data, mask):
    """
    展平特征数据，忽略 attention mask 之外的项目。
    """
    flat_data = data.view(-1, data.size(-1))
    attended_tokens = mask.view(-1).nonzero()[:, 0]
    return flat_data[attended_tokens]
