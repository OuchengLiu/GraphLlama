import os
# os.environ['CURL_CA_BUNDLE'] = ''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from torch.utils.data import Dataset as Dataset2
from torch.nn import CrossEntropyLoss

from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, TrainingArguments
from transformers import AutoTokenizer , AutoConfig, AutoModelForCausalLM

from typing import Optional
from peft import LoraConfig
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset, DatasetDict

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import GraphPropPredDataset
from ogb.nodeproppred import NodePropPredDataset


# from huggingface_hub import login
# login("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")


MODEL_NAME = "7B"  # "beomi/llama-2-ko-7b"  # "7B"  # "huggyllama/llama-7b"
ID2LABLE_FILE_PATH = "Data/labelidx2arxivcategeory.csv"
K = 5
NUM_LABELS = 40
MAX_LENGTH = 140
BATCH_SIZE = 64
EPOCHS = 30
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-3
LoRA_R = 64
LoRA_ALPHA = 8


@dataclass
class ScriptArguments:
    # Setting for Model
    model_name: Optional[str] = field(default=MODEL_NAME, metadata={"help": "the model name"})

    # Setting for Training
    learning_rate: Optional[float] = field(default=LEARNING_RATE, metadata={"help": "the learning rate"})  # 1.41e-5
    batch_size: Optional[int] = field(default=BATCH_SIZE, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=GRADIENT_ACCUMULATION_STEPS, metadata={"help": "the number of gradient accumulation steps"})
    num_train_epochs: Optional[int] = field(default=EPOCHS, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})

    # Setting for LoRA
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=LoRA_R, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=LoRA_ALPHA, metadata={"help": "the alpha parameter of the LoRA adapters"})
    
    # Setting for Save
    save_steps: Optional[int] = field(default=100, metadata={"help": "Number of updates steps before two checkpoint saves"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    output_dir: Optional[str] = field(default="Output", metadata={"help": "the output directory"})
    
    # Setting for Log
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})

    # Setting for Permission
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})

    # Maybe use for CausalLM
    # dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})  
    # seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"}) 


class CustomDataset(Dataset2):
    def __init__(self, embeds, labels):
        self.embeds = embeds.squeeze(1).to(torch.bfloat16)
        self.labels = labels
        self.attention_mask = create_attention_mask(self.embeds)

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        item = {"inputs_embeds": self.embeds[idx], "labels": self.labels[idx], "attention_mask": self.attention_mask[idx]}
        return item


def create_attention_mask(embeds):
    last_feature_zero = embeds[:, :, -1] == 0

    # 将结果转换为整数（0或1）
    attention_mask = (~last_feature_zero).to(torch.int)
    return attention_mask


def embed_extract(tokenizer, extract_embedding, text):
  tokens = tokenizer(text, return_tensors="pt")
  embeds = extract_embedding(tokens["input_ids"])
  return embeds


def expand_embedding(embeddings):
    # 检查是单个嵌入向量还是一批嵌入向量
    if len(embeddings.shape) == 1:
        # 处理单个 [1024] 维度的向量
        if embeddings.shape[0] != 1024:
            raise ValueError("Input embedding must be a 1D tensor with 1024 elements")
        return (embeddings / 4.0).repeat(4)
    elif len(embeddings.shape) == 2:
        # 处理 [N, 1024] 维度的批量向量
        if embeddings.shape[1] != 1024:
            raise ValueError("Each embedding in the batch must have 1024 elements")
        return (embeddings / 4.0).repeat(1, 4)
    else:
        raise ValueError("Input embedding must be either 1D or 2D")


def pad_tensor(tensor, pad_size, dim, pad_value):
    """将张量在指定维度的最左侧填充到指定的大小，并使用指定的填充值。"""
    # 确保填充尺寸不会是负数
    padding = max(pad_size - tensor.size(dim), 0)
    
    # 生成一个新的pad元组，用于在左侧填充
    pad = (0, 0) * (tensor.dim() - dim - 1) + (padding, 0)
    
    return torch.nn.functional.pad(tensor, pad, 'constant', pad_value)


def get_category_by_label_idx(df, label_idx):
    # 检查label_idx是否在数据中
    if label_idx not in df['label idx'].values:
        return "Label index not found"

    # 提取对应的arxiv category
    category = df[df['label idx'] == label_idx]['arxiv category'].iloc[0]

    return category.split()[-1].strip()


def construct_instruction(node_idx, node_feat, K, K_idx, K_feat, node_target, tokenizer, extract_embedding, df):
    query_part_1 = f"# Human: Central node [{node_idx}] is featured with text feature"
    query_part_2 = f"are the top-{K} similar nodes {K_idx}'s features within two-hops.\n"
    query_part_3 = f"Which category should central node [{node_idx}]  be classified as?\n# Assistant: "
    answer = get_category_by_label_idx(df, int(node_target))

    embed_1 = embed_extract(tokenizer, extract_embedding, query_part_1)
    embed_2 = embed_extract(tokenizer, extract_embedding, query_part_2)
    embed_3 = embed_extract(tokenizer, extract_embedding, query_part_3)
    embed_answer = embed_extract(tokenizer, extract_embedding, answer)

    node_feat = expand_embedding(node_feat).view(1, 1, -1)
    K_feat = expand_embedding(K_feat).view(1, K, -1)

    device = embed_1.device
    node_feat = node_feat.to(device)
    K_feat = K_feat.to(device)

    # 构造labels
    instrcution_embedding = torch.cat((embed_1, node_feat, K_feat, embed_2, embed_3), dim=1)
    tokens = tokenizer(answer, return_tensors="pt")
    answer_tokens = torch.tensor([-100] * instrcution_embedding.size(1)).view(1, -1)
    answer_tokens = torch.cat((answer_tokens, tokens['input_ids'][0, 1].view(1, -1)), dim=1)  # 一个奇怪的问题，tokenizer后会变成[[1, label_id]]，这个[1]==<s>不知从何而来

    # 如果需要补齐
    if answer_tokens.shape[1] < MAX_LENGTH:
        # 在第二维的末尾进行填充
        answer_tokens = pad_tensor(answer_tokens, pad_size=MAX_LENGTH, dim=1, pad_value=-100)

    # 构造inputs
    instrcution_embedding = torch.cat((embed_1, node_feat, K_feat, embed_2, embed_3, embed_answer), dim=1)
    if instrcution_embedding.size(1) < MAX_LENGTH:
        instrcution_embedding = pad_tensor(instrcution_embedding, pad_size=MAX_LENGTH, dim=1, pad_value=0)

    return instrcution_embedding, answer_tokens


def load_model():
    script_args = ScriptArguments()
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        device_map = "auto"  #{"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    # 加载预训练模型的配置
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"
    config.max_new_tokens = 2

    # 加载预训练模型，并添加分类头
    model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    config=config,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer, script_args


def load_data(x_embs_file, top_k_neighbors_file, dataset_type='train'):
    # 加载嵌入和邻居索引
    x_embs = torch.load(x_embs_file)
    # 使用 JSON 加载 top_k_neighbors 数据
    with open(top_k_neighbors_file, 'r') as file:
        top_k_neighbors = json.load(file)

    # 加载ogbn-arxiv数据集
    dataset = NodePropPredDataset(name='ogbn-arxiv')
    data,y = dataset[0]
    split_idx = dataset.get_idx_split()

    # 根据输入参数选择索引
    if dataset_type == 'train':
        idxs = split_idx['train']
    elif dataset_type == 'valid':
        idxs = split_idx['valid']
    elif dataset_type == 'test':
        idxs = split_idx['test']
    else:
        raise ValueError("Invalid dataset type. Choose 'train', 'valid', or 'test'.")

    data_list = []

    for position, idx in tqdm(enumerate(idxs), desc=f"Processing {dataset_type} Data"):
        node_idx = idx.item()  # 节点索引
        node_feat = x_embs[node_idx]  # 节点特征
        # 使用位置索引获取top_k_neighbors
        k_idx = top_k_neighbors[str(position)]   #top_k_neighbors[str(node_idx)]  # 使用位置索引
        k_feat = x_embs[k_idx]  # 邻居特征
        node_target = y[idx]  # 节点目标/标签

        data_list.append({
            'node_idx': node_idx,
            'node_feat': node_feat,
            'K_idx': k_idx,
            'K_feat': k_feat,
            'label': node_target
        })


    return data_list


def compute_metrics(pred):
    
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    # llama2_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # for i in range(len(predictions)):
    # #   print(len(llama2_tokenizer.convert_ids_to_tokens(predictions[i])))
    #   print(labels[i])
    #   print(predictions[i])
    #   print(llama2_tokenizer.decode(predictions[i]))
      
    
    # Calculate accuracy
    accuracy = accuracy_score(labels[ :, :, -1].squeeze(), preds[ :, -1])

   # Calculate precision, recall, and F1-score
    # precision = precision_score(labels, preds, average='weighted')
    # recall = recall_score(labels, preds, average='weighted')
    # f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        # 'precision': precision,
        # 'recall': recall,
        # 'f1': f1
    }


def main():
    llama2_model, llama2_tokenizer, script_args = load_model()
    extract_embedding = llama2_model.get_input_embeddings()

    # 使用函数生成数据
    embs_path = 'Data/x_embs.pt'
    train_top_k_neighbors_path = 'Data/train_top_k_neighbors.json'
    valid_top_k_neighbors_path = 'Data/valid_top_k_neighbors.json'
    test_top_k_neighbors_path='Data/test_top_k_neighbors.json'

    try:
    # 直接加载 .pt 文件
        train_data=torch.load("Instruction/cl_train_data.pt")
        valid_data=torch.load("Instruction/cl_valid_data.pt")
        test_data=torch.load("Instruction/cl_test_datapt")
    except FileNotFoundError:
    # 如果文件不存在，则处理数据并保存
        train_data = load_data(embs_path, train_top_k_neighbors_path, 'train')
        valid_data = load_data(embs_path, valid_top_k_neighbors_path, 'valid')
        test_data = load_data(embs_path, test_top_k_neighbors_path, 'test')
        torch.save(train_data, "Instruction/cl_train_data.pt")
        torch.save(valid_data, "Instruction/cl_valid_data.pt")
        torch.save(test_data, "Instruction/cl_test_data.pt")


    try:
    # 直接加载 .pt 文件
        train_instructions = torch.load("Instruction/cl_train_instructions.pt")
    except FileNotFoundError:
    # 如果文件不存在，则处理数据并保存
        # 读取CSV文件
        df = pd.read_csv(ID2LABLE_FILE_PATH)
        train_instructions = [construct_instruction(node['node_idx'], node['node_feat'], K, node['K_idx'], node['K_feat'], node['label'], llama2_tokenizer, extract_embedding, df)
                  for node in tqdm(train_data, desc='Processing Train Instructions')]
        torch.save(train_instructions, "Instruction/cl_train_instructions.pt")

    try:
    # 直接加载 .pt 文件
        valid_instructions = torch.load("Instruction/cl_valid_instructions.pt")
    except FileNotFoundError:
    # 如果文件不存在，则处理数据并保存
        valid_instructions = [construct_instruction(node['node_idx'], node['node_feat'], K, node['K_idx'], node['K_feat'], node['label'], llama2_tokenizer, extract_embedding, df)
                  for node in tqdm(valid_data, desc='Processing Validation Instructions')]
        torch.save(valid_instructions, "Instruction/cl_valid_instructions.pt")


    random.shuffle(train_instructions)
    random.shuffle(valid_instructions)

    train_embeds, train_labels = zip(*train_instructions)
    valid_embeds, valid_labels = zip(*valid_instructions)


    # !!!
    # 转换为 torch.tensor 并确保数据在 CPU 上
    train_embeds = torch.stack(train_embeds).cpu()
    # train_labels = torch.tensor(train_labels, dtype=torch.long).cpu()
    valid_embeds = torch.stack(valid_embeds).cpu()
    # valid_labels = torch.tensor(valid_labels, dtype=torch.long).cpu()

    # 创建 MyDataset 实例
    train_dataset = CustomDataset(train_embeds, train_labels)
    valid_dataset = CustomDataset(valid_embeds, valid_labels)


    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        evaluation_strategy = "epoch",
        # save_strategy = "epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
    )

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head'],  # Select LoRA tuning modules.
            bias="none",
            task_type= "CAUSAL_LM",  #"CAUSAL_LM", FEATURE_EXTRACTION, QUESTION_ANS, SEQ_2_SEQ_LM, SEQ_CLS, TOKEN_CLS"
        )
    else:
        peft_config = None

    llama2_model.add_adapter(peft_config)

    trainer = Trainer(
        model=llama2_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    metrics=trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()


