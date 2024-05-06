import argparse
import importlib
import json
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
import datetime

print_ = print

def lm_eval_fn(args):
    global print_
    lm_obj = HFLM(pretrained=args.model,tokenizer=args.tokenizer, batch_size="auto")
    task_manager = lm_eval.tasks.TaskManager()

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=args.tasks,
        num_fewshot=0,
        task_manager=task_manager,
    )
    with open('eval_origin.json', 'w') as f:
        json.dump(results, f)
    table_str = make_table(results)
    with open('eval_origin_table.txt', 'w') as f:
        json.dump(table_str, f)
    print_(table_str)

    from quantize import quantize
    from config.w8x8 import quantize_cfg
    quantize(lm_obj._model,cfg=quantize_cfg)
    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_obj,
        tasks=args.tasks,
        num_fewshot=0,
        task_manager=task_manager,
    )
    with open('eval_quant.json', 'w') as f:
        json.dump(results, f)
    table_str = make_table(results)
    with open('eval_quant_table.txt', 'w') as f:
        json.dump(table_str, f)
    print_(table_str)

def ppl_eval_fn(args):
    global print_
    def evaluate_perplexity(model, tokenizer,dataset):
        def _perplexity(nlls, n_samples, seqlen):
            return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))
        data = None
        if dataset == "wikitext":
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
        elif dataset =="ptb":
            data = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
            data = tokenizer(" ".join(data['sentence']), return_tensors="pt")
        elif dataset == "c4":
            data = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
            data = tokenizer(" ".join(data[:]['text']), return_tensors="pt")
        else:
            raise f"Not support ppl eval dataset:{dataset}"
        data = data.input_ids.to(model.device)
        seqlen = 2048
        model = model.eval()
        n_samples = data.numel() // seqlen

        nlls = []

        with tqdm.tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
            for i in progress_bar:
                start_index = i * seqlen
                end_index = (i + 1) * seqlen
                batch = data[:, start_index:end_index].to(model.device)
                with torch.no_grad():
                    logits = model(batch).logits
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = data[:, start_index:end_index][:, 1:]
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * seqlen
                nlls.append(neg_log_likelihood)

                curr_ppl = _perplexity(nlls, i + 1, seqlen)
                progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

        ppl = _perplexity(nlls, n_samples, seqlen)
        print_(f"Perplexity on {dataset}: {ppl.item()}")
        return ppl.item()
    for dataset in args.datasets:
        print_(f"\n-----------------begin test ppl on dataset {dataset}-------------------\n")
        evaluate_perplexity(args.model,args.tokenizer,dataset)

def run_test(args,title:str):
    global print_
    print_(f"\n-------------------------{title}-----------------------------\n")
    if "ppl" in args.tasks:
        ppl_eval_fn(args)
        args.tasks.remove("ppl")
    if len(args.tasks) != 0:
        lm_eval_fn(args)

def parse_args():
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model","-m", 
        type=str, default="./model/TinyLlama-1.1B-Chat-v1.0",
        help="path to model or hugging face model id"
    )
    parser.add_argument(
        "--output","-o",
        type=str,
        default=f"./result-{now.month:02}-{now.day:02}-{now.hour:02}:{now.minute:02}:{now.second:02}.log",
        help="where to save eval result",
    )
    parser.add_argument(
        "--datasets","-d",
        type=str,
        default="wikitext,c4,ptb",
        help=" the dataset used to eval perplexity",
    )
    parser.add_argument(
        "--tasks","-t",
        type=str,
        default="mmlu,ppl,lambada_openai,boolq,arc_easy,arc_challenge,piqa,winogrande",
        help="tasks parameter for lm-evaluation-harness",
    )
    parser.add_argument(
        "--act-path","-a",
        type=str,
        default="./act_scales/llama-2-7b.pt",
        help="path to act_scales",
    )
    parser.add_argument(
        "--quant","-q",
        type=str,
        default="./config/w8x8.py",
        help="path to quant config",
    )
    return parser.parse_args()


def main():
    import os
    os.chdir(os.path.dirname(__file__))
    args = parse_args()
    args.datasets = args.datasets.split(",")
    model_name = args.model.split("/")[-1]
    setattr(args,"tokenizer",AutoTokenizer.from_pretrained(args.model))
    setattr(args,"model",AutoModelForCausalLM.\
            from_pretrained(args.model,torch_dtype=torch.float16,device_map="auto"))
    args.model.eval()
    out_f = open(args.output,"w")
    def print_fn(*value:object,sep=" ",end="\n",file=None,flush=False):
        out_f.write(sep.join([str(v) for v in value])+end)
        print(*value,sep=sep,end=end,file=file,flush=flush)
    global print_
    print_ = print_fn
    args.tasks = args.tasks.split(",")
    flag = "ppl" not in args.tasks
    run_test(args,f"test {model_name}")
    args.tasks = args.tasks if flag else (args.tasks + ["ppl"])
    # quantize 
    model_cfg=args.model.model.config
    spec = importlib.util.spec_from_file_location("quant_cfg_module", args.quant)
    quant_cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant_cfg_module)
    quantize_cfg = quant_cfg_module.get(model_cfg,args.act_path)
    from quantize import quantize
    quantize(args.model,quantize_cfg)
    
    run_test(args,f"test quantized {model_name}")
    out_f.close()

if __name__ == "__main__":
    main()